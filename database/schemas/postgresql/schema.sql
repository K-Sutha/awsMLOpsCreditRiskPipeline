-- Credit Risk Platform - PostgreSQL Database Schema
-- Production-ready schema for loan applications and risk management

-- Create database
CREATE DATABASE credit_risk_platform;

-- Connect to the database
\c credit_risk_platform;

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create applications table (main loan applications)
CREATE TABLE applications (
    application_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    person_age INTEGER NOT NULL CHECK (person_age >= 18 AND person_age <= 100),
    person_income DECIMAL(12,2) NOT NULL CHECK (person_income >= 0),
    person_home_ownership VARCHAR(20) NOT NULL CHECK (person_home_ownership IN ('RENT', 'OWN', 'MORTGAGE', 'OTHER')),
    person_emp_length DECIMAL(5,2) NOT NULL CHECK (person_emp_length >= 0 AND person_emp_length <= 50),
    loan_intent VARCHAR(50) NOT NULL CHECK (loan_intent IN ('PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION')),
    loan_grade VARCHAR(5) NOT NULL CHECK (loan_grade IN ('A', 'B', 'C', 'D', 'E', 'F', 'G')),
    loan_amnt DECIMAL(12,2) NOT NULL CHECK (loan_amnt >= 0),
    loan_int_rate DECIMAL(5,2) NOT NULL CHECK (loan_int_rate >= 0 AND loan_int_rate <= 50),
    loan_percent_income DECIMAL(5,4) NOT NULL CHECK (loan_percent_income >= 0 AND loan_percent_income <= 1),
    cb_person_default_on_file VARCHAR(1) NOT NULL CHECK (cb_person_default_on_file IN ('Y', 'N')),
    cb_person_cred_hist_length INTEGER NOT NULL CHECK (cb_person_cred_hist_length >= 0),
    status VARCHAR(20) DEFAULT 'PENDING' CHECK (status IN ('PENDING', 'PROCESSING', 'APPROVED', 'REJECTED', 'MANUAL_REVIEW')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create risk_scores table (ML model predictions)
CREATE TABLE risk_scores (
    risk_score_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    application_id UUID NOT NULL REFERENCES applications(application_id) ON DELETE CASCADE,
    risk_score DECIMAL(5,4) NOT NULL CHECK (risk_score >= 0 AND risk_score <= 1),
    risk_category VARCHAR(20) NOT NULL CHECK (risk_category IN ('LOW', 'MEDIUM', 'HIGH')),
    model_version VARCHAR(10) NOT NULL,
    confidence_score DECIMAL(5,4) CHECK (confidence_score >= 0 AND confidence_score <= 1),
    feature_importance JSONB,
    prediction_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    processing_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create decisions table (loan approval/rejection decisions)
CREATE TABLE decisions (
    decision_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    application_id UUID NOT NULL REFERENCES applications(application_id) ON DELETE CASCADE,
    decision VARCHAR(20) NOT NULL CHECK (decision IN ('APPROVE', 'REJECT', 'MANUAL_REVIEW')),
    decision_reason TEXT,
    risk_score DECIMAL(5,4) NOT NULL,
    underwriter_id VARCHAR(50),
    underwriter_notes TEXT,
    decision_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create audit_logs table (compliance and tracking)
CREATE TABLE audit_logs (
    log_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    application_id UUID REFERENCES applications(application_id) ON DELETE SET NULL,
    action VARCHAR(50) NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    entity_id UUID,
    old_value JSONB,
    new_value JSONB,
    user_id VARCHAR(50),
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create model_performance table (ML model tracking)
CREATE TABLE model_performance (
    performance_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_version VARCHAR(10) NOT NULL,
    metric_name VARCHAR(50) NOT NULL,
    metric_value DECIMAL(10,6) NOT NULL,
    evaluation_date DATE NOT NULL,
    dataset_size INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create portfolio_analytics table (business metrics)
CREATE TABLE portfolio_analytics (
    analytics_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    date DATE NOT NULL,
    total_applications INTEGER NOT NULL DEFAULT 0,
    approved_applications INTEGER NOT NULL DEFAULT 0,
    rejected_applications INTEGER NOT NULL DEFAULT 0,
    manual_review_applications INTEGER NOT NULL DEFAULT 0,
    total_loan_amount DECIMAL(15,2) NOT NULL DEFAULT 0,
    approved_loan_amount DECIMAL(15,2) NOT NULL DEFAULT 0,
    average_risk_score DECIMAL(5,4),
    default_rate DECIMAL(5,4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX idx_applications_status ON applications(status);
CREATE INDEX idx_applications_created_at ON applications(created_at);
CREATE INDEX idx_applications_person_age ON applications(person_age);
CREATE INDEX idx_applications_person_income ON applications(person_income);
CREATE INDEX idx_applications_loan_grade ON applications(loan_grade);

CREATE INDEX idx_risk_scores_application_id ON risk_scores(application_id);
CREATE INDEX idx_risk_scores_model_version ON risk_scores(model_version);
CREATE INDEX idx_risk_scores_risk_score ON risk_scores(risk_score);
CREATE INDEX idx_risk_scores_created_at ON risk_scores(created_at);

CREATE INDEX idx_decisions_application_id ON decisions(application_id);
CREATE INDEX idx_decisions_decision ON decisions(decision);
CREATE INDEX idx_decisions_underwriter_id ON decisions(underwriter_id);
CREATE INDEX idx_decisions_created_at ON decisions(created_at);

CREATE INDEX idx_audit_logs_application_id ON audit_logs(application_id);
CREATE INDEX idx_audit_logs_action ON audit_logs(action);
CREATE INDEX idx_audit_logs_created_at ON audit_logs(created_at);

CREATE INDEX idx_model_performance_model_version ON model_performance(model_version);
CREATE INDEX idx_model_performance_evaluation_date ON model_performance(evaluation_date);

CREATE INDEX idx_portfolio_analytics_date ON portfolio_analytics(date);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_applications_updated_at 
    BEFORE UPDATE ON applications 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create audit trigger function
CREATE OR REPLACE FUNCTION audit_trigger_function()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        INSERT INTO audit_logs (action, entity_type, entity_id, old_value)
        VALUES (TG_OP, TG_TABLE_NAME, OLD.application_id, to_jsonb(OLD));
        RETURN OLD;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO audit_logs (action, entity_type, entity_id, old_value, new_value)
        VALUES (TG_OP, TG_TABLE_NAME, NEW.application_id, to_jsonb(OLD), to_jsonb(NEW));
        RETURN NEW;
    ELSIF TG_OP = 'INSERT' THEN
        INSERT INTO audit_logs (action, entity_type, entity_id, new_value)
        VALUES (TG_OP, TG_TABLE_NAME, NEW.application_id, to_jsonb(NEW));
        RETURN NEW;
    END IF;
    RETURN NULL;
END;
$$ language 'plpgsql';

-- Create audit triggers
CREATE TRIGGER audit_applications_trigger
    AFTER INSERT OR UPDATE OR DELETE ON applications
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

CREATE TRIGGER audit_decisions_trigger
    AFTER INSERT OR UPDATE OR DELETE ON decisions
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

-- Create views for common queries
CREATE VIEW application_summary AS
SELECT 
    a.application_id,
    a.person_age,
    a.person_income,
    a.loan_amnt,
    a.loan_grade,
    a.status,
    rs.risk_score,
    rs.risk_category,
    d.decision,
    d.decision_reason,
    a.created_at
FROM applications a
LEFT JOIN risk_scores rs ON a.application_id = rs.application_id
LEFT JOIN decisions d ON a.application_id = d.application_id;

-- Create view for portfolio metrics
CREATE VIEW daily_portfolio_metrics AS
SELECT 
    DATE(created_at) as date,
    COUNT(*) as total_applications,
    COUNT(CASE WHEN status = 'APPROVED' THEN 1 END) as approved_count,
    COUNT(CASE WHEN status = 'REJECTED' THEN 1 END) as rejected_count,
    AVG(loan_amnt) as avg_loan_amount,
    SUM(loan_amnt) as total_loan_amount
FROM applications
GROUP BY DATE(created_at)
ORDER BY date DESC;

-- Grant permissions (adjust as needed for your setup)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO credit_risk_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO credit_risk_user;

"""
DataQA End-to-End Usage Examples

This module demonstrates complete workflows and real-world usage patterns
for the DataQA framework, from setup to advanced analytics.
"""

import asyncio
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import json
import time

from dataqa import (
    DataQAClient,
    DataAgent,
    AgentConfig,
    Document,
    create_agent,
    agent_session,
    quick_query
)


def example_1_sales_analytics_workflow():
    """
    Example 1: Complete Sales Analytics Workflow
    
    This example demonstrates a typical business analytics workflow:
    1. Set up agent with business context
    2. Load and explore data
    3. Perform analysis
    4. Generate visualizations
    5. Create summary report
    """
    print("=== Example 1: Sales Analytics Workflow ===")
    
    # Step 1: Create agent with analytics configuration
    config = {
        "name": "sales-analytics-agent",
        "description": "Agent specialized for sales data analysis",
        "llm": {
            "provider": "openai",
            "model": "gpt-4",
            "api_key": "${OPENAI_API_KEY}",
            "temperature": 0.1
        },
        "knowledge": {
            "provider": "faiss",
            "embedding_model": "all-mpnet-base-v2",
            "top_k": 8
        },
        "executor": {
            "provider": "inmemory",
            "database_type": "duckdb",
            "max_execution_time": 120.0,
            "max_memory_mb": 2048
        },
        "workflow": {
            "strategy": "plan_execute",
            "require_approval": True,
            "enable_visualization": True,
            "conversation_memory": True
        }
    }
    
    with agent_session("sales-agent", config=config) as agent:
        # Step 2: Add business context and schema knowledge
        business_knowledge = [
            Document(
                content="""
                Sales Database Schema:
                - sales_transactions: id, date, product_id, customer_id, amount, quantity, sales_rep_id
                - products: id, name, category, price, cost, launch_date
                - customers: id, name, email, segment, region, signup_date
                - sales_reps: id, name, territory, hire_date, quota
                
                Business Rules:
                - Customer segments: Premium (>$10k annual), Standard ($1k-$10k), Basic (<$1k)
                - Territories: North, South, East, West
                - Q4 typically shows 40% higher sales due to holiday season
                """,
                metadata={"type": "schema", "importance": "high"},
                source="sales_schema.md"
            ),
            Document(
                content="""
                Key Performance Indicators:
                - Monthly Recurring Revenue (MRR)
                - Customer Acquisition Cost (CAC)
                - Customer Lifetime Value (CLV)
                - Churn Rate
                - Average Order Value (AOV)
                - Sales Rep Performance vs Quota
                """,
                metadata={"type": "kpi", "importance": "high"},
                source="kpi_definitions.md"
            ),
            Document(
                content="""
                Recent Business Context:
                - New product line launched in Q3 2024
                - Marketing campaign ran from Sept-Nov 2024
                - Sales team expanded by 30% in Q2 2024
                - Price increase of 15% implemented in August 2024
                """,
                metadata={"type": "context", "period": "2024"},
                source="business_updates.md"
            )
        ]
        
        print("Ingesting business knowledge...")
        agent.ingest_knowledge(business_knowledge)
        
        # Step 3: Data exploration and validation
        print("\n--- Data Exploration ---")
        
        queries = [
            "First, let me understand our data. Show me the structure and size of our main tables",
            "What's the date range of our sales data?",
            "How many unique customers, products, and sales reps do we have?"
        ]
        
        for query in queries:
            print(f"\nQuery: {query}")
            response = agent.query(query)
            print(f"Response: {response}")
            
            # Handle approval if needed
            if agent.has_pending_approval():
                print("Operation requires approval. Auto-approving for demo...")
                agent.approve_operation(approved=True)
        
        # Step 4: Core business analysis
        print("\n--- Core Business Analysis ---")
        
        analysis_queries = [
            "Calculate our key metrics: total revenue, number of orders, average order value for the last 12 months",
            "Show me monthly sales trends for 2024. Include both revenue and order count",
            "Analyze customer segments - how much revenue comes from each segment?",
            "Which are our top 10 products by revenue? Include their categories",
            "How are our sales reps performing against their quotas?",
            "What's our customer retention rate and churn analysis?"
        ]
        
        analysis_results = {}
        for i, query in enumerate(analysis_queries, 1):
            print(f"\nAnalysis {i}: {query}")
            response = agent.query(query)
            print(f"Result: {response}")
            analysis_results[f"analysis_{i}"] = response
            
            # Auto-approve for demo
            if agent.has_pending_approval():
                agent.approve_operation(approved=True)
        
        # Step 5: Advanced analytics and insights
        print("\n--- Advanced Analytics ---")
        
        advanced_queries = [
            "Create a comprehensive sales dashboard with multiple visualizations showing trends, segments, and performance",
            "Perform cohort analysis to understand customer behavior over time",
            "Identify seasonal patterns and forecast next quarter's sales",
            "Find correlations between product categories and customer segments"
        ]
        
        for query in advanced_queries:
            print(f"\nAdvanced Query: {query}")
            response = agent.query(query)
            print(f"Insight: {response}")
            
            if agent.has_pending_approval():
                agent.approve_operation(approved=True)
        
        # Step 6: Generate executive summary
        print("\n--- Executive Summary ---")
        summary_query = """
        Based on all our analysis, create an executive summary report that includes:
        1. Key performance metrics and trends
        2. Top insights and findings
        3. Recommendations for business improvement
        4. Areas of concern or opportunity
        Format it as a professional business report.
        """
        
        print("Generating executive summary...")
        summary = agent.query(summary_query)
        print(f"\nExecutive Summary:\n{summary}")
        
        if agent.has_pending_approval():
            agent.approve_operation(approved=True)
        
        # Step 7: Save results
        print("\n--- Saving Results ---")
        conversation_history = agent.get_conversation_history()
        
        # Save conversation and results
        results = {
            "workflow": "sales_analytics",
            "timestamp": time.time(),
            "analysis_results": analysis_results,
            "executive_summary": summary,
            "conversation_length": len(conversation_history)
        }
        
        with open("sales_analytics_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print("Sales analytics workflow completed successfully!")
        print(f"Results saved to sales_analytics_results.json")


async def example_2_multi_agent_collaboration():
    """
    Example 2: Multi-Agent Collaboration
    
    Demonstrates how multiple specialized agents can work together
    on a complex business problem.
    """
    print("\n=== Example 2: Multi-Agent Collaboration ===")
    
    async with DataQAClient() as client:
        # Create specialized agents
        agents_config = {
            "data-engineer": {
                "name": "data-engineer",
                "description": "Specialized in data quality and ETL operations",
                "llm": {"provider": "openai", "model": "gpt-4"},
                "workflow": {"strategy": "workflow", "require_approval": False}
            },
            "business-analyst": {
                "name": "business-analyst", 
                "description": "Focused on business metrics and KPI analysis",
                "llm": {"provider": "openai", "model": "gpt-4"},
                "workflow": {"strategy": "react", "enable_visualization": True}
            },
            "data-scientist": {
                "name": "data-scientist",
                "description": "Advanced analytics and machine learning",
                "llm": {"provider": "openai", "model": "gpt-4"},
                "executor": {"max_execution_time": 300.0, "max_memory_mb": 4096}
            }
        }
        
        # Create all agents
        agents = {}
        for name, config in agents_config.items():
            agent = await client.create_agent_async(name, config=config)
            agents[name] = agent
            print(f"Created {name} agent")
        
        # Collaborative workflow: Customer Churn Analysis
        print("\n--- Collaborative Workflow: Customer Churn Analysis ---")
        
        # Step 1: Data Engineer - Data Quality Assessment
        print("\n1. Data Engineer: Assessing data quality...")
        de_query = """
        Perform a comprehensive data quality assessment:
        1. Check for missing values in customer and transaction data
        2. Identify duplicate records
        3. Validate data types and ranges
        4. Check for referential integrity between tables
        5. Summarize data quality issues and recommendations
        """
        
        de_response = await client.query_async("data-engineer", de_query)
        print(f"Data Quality Report: {de_response}")
        
        # Step 2: Business Analyst - Customer Behavior Analysis
        print("\n2. Business Analyst: Analyzing customer behavior...")
        ba_query = """
        Analyze customer behavior patterns:
        1. Calculate customer lifetime value (CLV) distribution
        2. Identify purchase frequency patterns
        3. Analyze seasonal buying behavior
        4. Segment customers by engagement level
        5. Create visualizations showing customer journey stages
        """
        
        ba_response = await client.query_async("business-analyst", ba_query)
        print(f"Customer Behavior Analysis: {ba_response}")
        
        # Step 3: Data Scientist - Predictive Modeling
        print("\n3. Data Scientist: Building churn prediction model...")
        ds_query = """
        Build a customer churn prediction model:
        1. Feature engineering from customer transaction data
        2. Create training dataset with churn labels
        3. Train multiple models (logistic regression, random forest, gradient boosting)
        4. Evaluate model performance and select best model
        5. Identify top features that predict churn
        6. Generate churn risk scores for active customers
        """
        
        ds_response = await client.query_async("data-scientist", ds_query)
        print(f"Churn Prediction Model: {ds_response}")
        
        # Step 4: Business Analyst - Actionable Insights
        print("\n4. Business Analyst: Generating actionable insights...")
        insights_query = f"""
        Based on the data quality report and churn model results, provide:
        1. Key drivers of customer churn
        2. Customer segments at highest risk
        3. Recommended retention strategies
        4. Expected impact of interventions
        5. Implementation roadmap
        
        Data Quality Context: {de_response[:500]}...
        Model Results Context: {ds_response[:500]}...
        """
        
        insights_response = await client.query_async("business-analyst", insights_query)
        print(f"Actionable Insights: {insights_response}")
        
        # Step 5: Consolidate results
        print("\n--- Consolidating Multi-Agent Results ---")
        
        consolidated_results = {
            "workflow": "multi_agent_churn_analysis",
            "timestamp": time.time(),
            "agents_used": list(agents_config.keys()),
            "results": {
                "data_quality": de_response,
                "customer_behavior": ba_response,
                "churn_model": ds_response,
                "actionable_insights": insights_response
            }
        }
        
        with open("multi_agent_churn_analysis.json", "w") as f:
            json.dump(consolidated_results, f, indent=2)
        
        print("Multi-agent collaboration completed successfully!")


def example_3_real_time_monitoring_dashboard():
    """
    Example 3: Real-time Monitoring Dashboard
    
    Demonstrates how to use DataQA for continuous monitoring
    and alerting scenarios.
    """
    print("\n=== Example 3: Real-time Monitoring Dashboard ===")
    
    # Configuration for monitoring agent
    monitoring_config = {
        "name": "monitoring-agent",
        "description": "Real-time business metrics monitoring",
        "llm": {
            "provider": "openai",
            "model": "gpt-3.5-turbo",  # Faster for real-time use
            "temperature": 0.05
        },
        "executor": {
            "provider": "inmemory",
            "max_execution_time": 30.0  # Quick responses
        },
        "workflow": {
            "require_approval": False,  # Auto-execute for monitoring
            "enable_visualization": True
        }
    }
    
    with agent_session("monitor", config=monitoring_config) as agent:
        # Add monitoring knowledge
        monitoring_knowledge = [
            Document(
                content="""
                Key Business Metrics Thresholds:
                - Daily Revenue: Alert if <$50k or >$200k (unusual)
                - Order Volume: Alert if <100 or >1000 orders/day
                - Average Order Value: Alert if <$45 or >$150
                - Customer Acquisition: Alert if <10 new customers/day
                - Error Rate: Alert if >5% of transactions fail
                - Response Time: Alert if >2 seconds average
                """,
                metadata={"type": "thresholds", "importance": "critical"},
                source="monitoring_thresholds.md"
            ),
            Document(
                content="""
                Monitoring Schedule:
                - Revenue metrics: Every 15 minutes
                - Order processing: Every 5 minutes  
                - Customer metrics: Every hour
                - System health: Every minute
                - Weekly trends: Every Monday 9 AM
                - Monthly reports: 1st of each month
                """,
                metadata={"type": "schedule"},
                source="monitoring_schedule.md"
            )
        ]
        
        agent.ingest_knowledge(monitoring_knowledge)
        
        # Simulate real-time monitoring cycle
        print("Starting monitoring cycle...")
        
        monitoring_queries = [
            # Real-time health checks
            "Check current system health: revenue, orders, and error rates for the last hour",
            
            # Performance monitoring
            "Analyze today's performance vs yesterday and last week same day",
            
            # Anomaly detection
            "Identify any unusual patterns or anomalies in today's data",
            
            # Trend analysis
            "Show key trend indicators and whether we're on track for monthly goals",
            
            # Alert generation
            "Generate alerts for any metrics outside normal thresholds",
            
            # Predictive insights
            "Based on current trends, forecast end-of-day performance"
        ]
        
        monitoring_results = {}
        
        for i, query in enumerate(monitoring_queries, 1):
            print(f"\nMonitoring Check {i}: {query}")
            
            start_time = time.time()
            response = agent.query(query)
            response_time = time.time() - start_time
            
            print(f"Response ({response_time:.2f}s): {response}")
            
            monitoring_results[f"check_{i}"] = {
                "query": query,
                "response": response,
                "response_time": response_time,
                "timestamp": time.time()
            }
        
        # Generate monitoring dashboard
        print("\n--- Generating Monitoring Dashboard ---")
        dashboard_query = """
        Create a comprehensive monitoring dashboard that includes:
        1. Current status indicators (green/yellow/red)
        2. Key metrics with trend arrows
        3. Alert summary
        4. Performance charts
        5. Recommendations for immediate action
        Format as a structured dashboard report.
        """
        
        dashboard = agent.query(dashboard_query)
        print(f"\nMonitoring Dashboard:\n{dashboard}")
        
        # Save monitoring results
        monitoring_summary = {
            "workflow": "real_time_monitoring",
            "timestamp": time.time(),
            "monitoring_cycle_results": monitoring_results,
            "dashboard": dashboard,
            "total_checks": len(monitoring_queries),
            "avg_response_time": sum(r["response_time"] for r in monitoring_results.values()) / len(monitoring_results)
        }
        
        with open("monitoring_dashboard.json", "w") as f:
            json.dump(monitoring_summary, f, indent=2)
        
        print("Real-time monitoring cycle completed!")
        print(f"Average response time: {monitoring_summary['avg_response_time']:.2f}s")


def example_4_customer_journey_analysis():
    """
    Example 4: Customer Journey Analysis
    
    Comprehensive analysis of customer lifecycle from acquisition
    to retention, including cohort analysis and lifetime value.
    """
    print("\n=== Example 4: Customer Journey Analysis ===")
    
    # Customer analytics configuration
    config = AgentConfig(
        name="customer-journey-agent",
        description="Specialized agent for customer lifecycle analysis",
        llm={
            "provider": "openai",
            "model": "gpt-4",
            "temperature": 0.1,
            "max_tokens": 2000
        },
        knowledge={
            "provider": "faiss",
            "embedding_model": "all-mpnet-base-v2",
            "top_k": 10
        },
        executor={
            "provider": "inmemory",
            "max_execution_time": 180.0,
            "max_memory_mb": 3072,
            "allowed_imports": [
                "pandas", "numpy", "matplotlib", "seaborn", 
                "datetime", "scipy", "sklearn"
            ]
        },
        workflow={
            "strategy": "plan_execute",
            "max_iterations": 15,
            "require_approval": True,
            "enable_visualization": True,
            "conversation_memory": True
        }
    )
    
    agent = DataAgent(config)
    
    try:
        # Add customer journey knowledge
        journey_knowledge = [
            Document(
                content="""
                Customer Journey Stages:
                1. Awareness: First interaction with brand/product
                2. Consideration: Evaluating options, comparing features
                3. Purchase: First transaction completed
                4. Onboarding: First 30 days after purchase
                5. Engagement: Regular usage/interaction period
                6. Retention: Continued loyalty and repeat purchases
                7. Advocacy: Referrals and positive reviews
                8. Churn: Inactive for 90+ days
                """,
                metadata={"type": "journey_stages"},
                source="customer_journey.md"
            ),
            Document(
                content="""
                Customer Lifecycle Metrics:
                - Time to First Purchase (TTFP)
                - Customer Acquisition Cost (CAC)
                - Customer Lifetime Value (CLV)
                - Monthly Recurring Revenue (MRR)
                - Churn Rate by Cohort
                - Net Promoter Score (NPS)
                - Engagement Score
                - Retention Rate by Period
                """,
                metadata={"type": "metrics"},
                source="lifecycle_metrics.md"
            )
        ]
        
        agent.ingest_knowledge(journey_knowledge)
        
        # Customer journey analysis workflow
        journey_analyses = [
            # Acquisition Analysis
            {
                "stage": "Acquisition",
                "query": """
                Analyze customer acquisition patterns:
                1. Monthly acquisition trends over the last 2 years
                2. Acquisition channels and their effectiveness
                3. Cost per acquisition by channel
                4. Seasonal patterns in customer acquisition
                5. Create visualizations showing acquisition funnel
                """
            },
            
            # Cohort Analysis
            {
                "stage": "Cohort Analysis", 
                "query": """
                Perform comprehensive cohort analysis:
                1. Create monthly cohort tables showing retention rates
                2. Calculate cohort lifetime value progression
                3. Identify best and worst performing cohorts
                4. Analyze cohort behavior patterns
                5. Visualize cohort retention heatmaps
                """
            },
            
            # Engagement Analysis
            {
                "stage": "Engagement",
                "query": """
                Analyze customer engagement patterns:
                1. Define engagement metrics (frequency, recency, monetary)
                2. Segment customers by engagement level
                3. Track engagement evolution over customer lifetime
                4. Identify engagement drop-off points
                5. Create engagement scoring model
                """
            },
            
            # Lifetime Value Analysis
            {
                "stage": "Lifetime Value",
                "query": """
                Calculate and analyze customer lifetime value:
                1. Historical CLV calculation by customer segment
                2. Predictive CLV modeling
                3. CLV distribution analysis
                4. Factors that drive higher CLV
                5. CLV vs CAC ratio analysis
                """
            },
            
            # Churn Analysis
            {
                "stage": "Churn Analysis",
                "query": """
                Comprehensive churn analysis:
                1. Calculate churn rates by different time periods
                2. Identify early warning signals of churn
                3. Analyze churn reasons and patterns
                4. Customer win-back analysis
                5. Churn prevention recommendations
                """
            }
        ]
        
        journey_results = {}
        
        for analysis in journey_analyses:
            stage = analysis["stage"]
            query = analysis["query"]
            
            print(f"\n--- {stage} Analysis ---")
            print(f"Query: {query}")
            
            response = agent.query(query)
            print(f"Analysis Result: {response}")
            
            journey_results[stage.lower().replace(" ", "_")] = {
                "query": query,
                "response": response,
                "timestamp": time.time()
            }
            
            # Handle approval
            if agent.has_pending_approval():
                print("Analysis requires approval. Auto-approving for demo...")
                agent.approve_operation(approved=True)
        
        # Comprehensive journey insights
        print("\n--- Comprehensive Journey Insights ---")
        insights_query = """
        Based on all the customer journey analyses performed, provide:
        1. Key insights about our customer lifecycle
        2. Strengths and weaknesses in each journey stage
        3. Opportunities for improvement
        4. Recommended actions with expected impact
        5. Metrics to track for ongoing optimization
        
        Create a comprehensive customer journey optimization report.
        """
        
        insights = agent.query(insights_query)
        print(f"Journey Insights: {insights}")
        
        if agent.has_pending_approval():
            agent.approve_operation(approved=True)
        
        # Save comprehensive results
        final_results = {
            "workflow": "customer_journey_analysis",
            "timestamp": time.time(),
            "analyses_performed": list(journey_results.keys()),
            "detailed_results": journey_results,
            "comprehensive_insights": insights,
            "agent_info": agent.get_agent_info()
        }
        
        with open("customer_journey_analysis.json", "w") as f:
            json.dump(final_results, f, indent=2)
        
        print("Customer journey analysis completed successfully!")
        print(f"Analyzed {len(journey_analyses)} journey stages")
        
    finally:
        agent.shutdown()


def example_5_automated_reporting_pipeline():
    """
    Example 5: Automated Reporting Pipeline
    
    Demonstrates how to create automated, scheduled reports
    with DataQA for regular business reporting.
    """
    print("\n=== Example 5: Automated Reporting Pipeline ===")
    
    # Quick setup for different report types
    report_configs = {
        "daily": {
            "name": "daily-reporter",
            "llm": {"provider": "openai", "model": "gpt-3.5-turbo"},
            "workflow": {"require_approval": False, "enable_visualization": True}
        },
        "weekly": {
            "name": "weekly-reporter", 
            "llm": {"provider": "openai", "model": "gpt-4"},
            "workflow": {"strategy": "plan_execute", "enable_visualization": True}
        },
        "monthly": {
            "name": "monthly-reporter",
            "llm": {"provider": "openai", "model": "gpt-4"},
            "executor": {"max_execution_time": 300.0},
            "workflow": {"strategy": "plan_execute", "enable_visualization": True}
        }
    }
    
    # Report templates
    report_templates = {
        "daily": """
        Generate a daily business report including:
        1. Yesterday's key metrics (revenue, orders, customers)
        2. Comparison to previous day and week ago
        3. Top performing products and categories
        4. Any notable anomalies or issues
        5. Brief summary with 2-3 key takeaways
        Keep it concise and actionable.
        """,
        
        "weekly": """
        Create a comprehensive weekly business review:
        1. Week-over-week performance analysis
        2. Customer acquisition and retention metrics
        3. Product performance deep dive
        4. Sales team performance summary
        5. Key trends and pattern identification
        6. Recommendations for the upcoming week
        Include relevant charts and visualizations.
        """,
        
        "monthly": """
        Prepare a detailed monthly business report:
        1. Month-over-month growth analysis
        2. Customer cohort and lifetime value analysis
        3. Product portfolio performance review
        4. Market trends and competitive analysis
        5. Financial performance vs targets
        6. Strategic recommendations for next month
        7. Risk assessment and mitigation strategies
        Include executive summary and detailed appendices.
        """
    }
    
    # Generate all reports
    report_results = {}
    
    for report_type, template in report_templates.items():
        print(f"\n--- Generating {report_type.title()} Report ---")
        
        # Use quick_query for simplified report generation
        start_time = time.time()
        
        try:
            report = quick_query(
                template,
                agent_name=f"{report_type}-reporter",
                **report_configs[report_type]
            )
            
            generation_time = time.time() - start_time
            
            print(f"{report_type.title()} Report Generated ({generation_time:.2f}s):")
            print(f"{report[:500]}..." if len(report) > 500 else report)
            
            report_results[report_type] = {
                "content": report,
                "generation_time": generation_time,
                "timestamp": time.time(),
                "template_used": template
            }
            
        except Exception as e:
            print(f"Error generating {report_type} report: {e}")
            report_results[report_type] = {
                "error": str(e),
                "timestamp": time.time()
            }
    
    # Create report summary
    print("\n--- Report Generation Summary ---")
    
    successful_reports = [k for k, v in report_results.items() if "error" not in v]
    failed_reports = [k for k, v in report_results.items() if "error" in v]
    
    total_time = sum(r.get("generation_time", 0) for r in report_results.values())
    
    summary = {
        "pipeline": "automated_reporting",
        "timestamp": time.time(),
        "reports_generated": len(successful_reports),
        "reports_failed": len(failed_reports),
        "successful_reports": successful_reports,
        "failed_reports": failed_reports,
        "total_generation_time": total_time,
        "detailed_results": report_results
    }
    
    print(f"Successfully generated: {successful_reports}")
    if failed_reports:
        print(f"Failed to generate: {failed_reports}")
    print(f"Total generation time: {total_time:.2f}s")
    
    # Save all reports
    with open("automated_reports.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Save individual reports as text files
    for report_type, result in report_results.items():
        if "content" in result:
            with open(f"{report_type}_report.txt", "w") as f:
                f.write(f"# {report_type.title()} Business Report\n")
                f.write(f"Generated: {time.ctime(result['timestamp'])}\n")
                f.write(f"Generation Time: {result['generation_time']:.2f}s\n\n")
                f.write(result["content"])
    
    print("Automated reporting pipeline completed!")


def main():
    """Run all end-to-end examples."""
    print("DataQA End-to-End Usage Examples")
    print("=" * 60)
    
    try:
        # Run synchronous examples
        example_1_sales_analytics_workflow()
        example_3_real_time_monitoring_dashboard()
        example_4_customer_journey_analysis()
        example_5_automated_reporting_pipeline()
        
        # Run asynchronous examples
        print("\nRunning asynchronous examples...")
        asyncio.run(example_2_multi_agent_collaboration())
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("All end-to-end examples completed!")
    print("\nGenerated files:")
    print("- sales_analytics_results.json")
    print("- multi_agent_churn_analysis.json") 
    print("- monitoring_dashboard.json")
    print("- customer_journey_analysis.json")
    print("- automated_reports.json")
    print("- daily_report.txt")
    print("- weekly_report.txt")
    print("- monthly_report.txt")


if __name__ == "__main__":
    main()
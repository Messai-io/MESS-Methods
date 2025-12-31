"""
Model Failure Analysis
Diagnose why current models are performing poorly and identify solutions
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import structlog

logger = structlog.get_logger()


class ModelFailureAnalyzer:
    """Analyze why current ML models are failing"""
    
    def __init__(self, data_dir: str = "data", results_dir: str = "models/honest_baseline"):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
    
    def load_data_and_results(self):
        """Load current data and model results"""
        # Load clean data
        data_file = self.data_dir / "ultimate_with_mp" / "ultimate_ml_dataset.csv"
        if not data_file.exists():
            data_file = self.data_dir / "ultimate_combined" / "ultimate_training_data.csv"
        
        df = pd.read_csv(data_file)
        
        # Remove synthetic MP data
        df = df[~df['data_source'].str.contains('MP_', na=False)].copy()
        
        # Load results
        results_file = self.results_dir / "honest_baseline_results.json"
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        return df, results
    
    def analyze_data_distribution(self, df):
        """Analyze data distribution problems"""
        logger.info("Analyzing data distribution issues...")
        
        analysis = {}
        
        # Power density distribution
        power_stats = {
            'min': float(df['power_density'].min()),
            'max': float(df['power_density'].max()),
            'mean': float(df['power_density'].mean()),
            'median': float(df['power_density'].median()),
            'std': float(df['power_density'].std()),
            'skewness': float(df['power_density'].skew()),
            'range_ratio': float(df['power_density'].max() / df['power_density'].min())
        }
        
        analysis['power_distribution'] = power_stats
        
        # System type imbalance
        system_counts = df['data_source'].value_counts().to_dict()
        total_samples = len(df)
        system_percentages = {k: (v/total_samples)*100 for k, v in system_counts.items()}
        
        analysis['system_imbalance'] = {
            'counts': system_counts,
            'percentages': system_percentages,
            'dominant_system': max(system_percentages.items(), key=lambda x: x[1]),
            'imbalance_ratio': max(system_counts.values()) / min(system_counts.values())
        }
        
        # Feature correlation issues
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            high_corr_pairs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.9:
                        high_corr_pairs.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': corr_val
                        })
            
            analysis['feature_correlation'] = {
                'high_correlation_pairs': high_corr_pairs,
                'max_correlation': float(corr_matrix.abs().max().max()),
                'problematic_features': len(high_corr_pairs)
            }
        
        # Data sparsity issues
        missing_data = df.isnull().sum()
        analysis['data_quality'] = {
            'missing_values': missing_data.to_dict(),
            'complete_samples': int((~df.isnull().any(axis=1)).sum()),
            'completion_rate': float((~df.isnull().any(axis=1)).mean())
        }
        
        return analysis
    
    def analyze_feature_problems(self, df):
        """Analyze feature engineering problems"""
        logger.info("Analyzing feature problems...")
        
        problems = []
        
        # Check for redundant temperature features
        temp_features = [col for col in df.columns if 'temp' in col.lower()]
        if len(temp_features) > 2:
            problems.append({
                'type': 'redundant_features',
                'description': f'Multiple temperature features: {temp_features}',
                'impact': 'Multicollinearity reduces model stability'
            })
        
        # Check for low-variance features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        low_variance_features = []
        for col in numeric_cols:
            if df[col].var() < 0.01:
                low_variance_features.append(col)
        
        if low_variance_features:
            problems.append({
                'type': 'low_variance',
                'description': f'Low variance features: {low_variance_features}',
                'impact': 'Features provide little predictive information'
            })
        
        # Check for constant features
        constant_features = []
        for col in numeric_cols:
            if df[col].nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            problems.append({
                'type': 'constant_features',
                'description': f'Constant features: {constant_features}',
                'impact': 'No predictive value, should be removed'
            })
        
        # Check for interaction features with limited cases
        interaction_features = [col for col in df.columns if '_interaction' in col]
        for col in interaction_features:
            non_zero_count = (df[col] != 0).sum()
            if non_zero_count < 0.1 * len(df):
                problems.append({
                    'type': 'sparse_interaction',
                    'description': f'{col}: only {non_zero_count} non-zero values',
                    'impact': 'Interaction rarely activated, limited usefulness'
                })
        
        return problems
    
    def identify_core_issues(self, df, results, data_analysis):
        """Identify the core reasons for model failure"""
        logger.info("Identifying core failure reasons...")
        
        issues = []
        
        # Issue 1: Extreme data imbalance
        imbalance_ratio = data_analysis['system_imbalance']['imbalance_ratio']
        if imbalance_ratio > 10:
            issues.append({
                'severity': 'CRITICAL',
                'issue': 'Severe Class Imbalance',
                'description': f'Data imbalance ratio: {imbalance_ratio:.1f}:1',
                'impact': 'Model optimizes for majority class (batteries), fails on others',
                'solution': 'Collect more MFC/fuel cell data or use stratified sampling'
            })
        
        # Issue 2: Extreme power density range
        range_ratio = data_analysis['power_distribution']['range_ratio']
        if range_ratio > 1000:
            issues.append({
                'severity': 'CRITICAL', 
                'issue': 'Extreme Target Range',
                'description': f'Power density range: {range_ratio:.0f}:1 ratio',
                'impact': 'Single model cannot handle such extreme range variations',
                'solution': 'Log transform target or separate models by power range'
            })
        
        # Issue 3: Insufficient predictive features
        feature_count = len([col for col in df.columns if col not in ['power_density', 'data_source']])
        if feature_count < 10:
            issues.append({
                'severity': 'HIGH',
                'issue': 'Insufficient Features',
                'description': f'Only {feature_count} predictive features available',
                'impact': 'Not enough information to make accurate predictions',
                'solution': 'Add materials properties, operating conditions, design parameters'
            })
        
        # Issue 4: No domain-specific features
        materials_features = [col for col in df.columns if any(mat in col.lower() 
                            for mat in ['material', 'electrode', 'catalyst', 'electrolyte'])]
        if len(materials_features) == 0:
            issues.append({
                'severity': 'HIGH',
                'issue': 'No Materials Information',
                'description': 'No electrode, catalyst, or electrolyte features',
                'impact': 'Cannot account for material-dependent performance',
                'solution': 'Add materials database integration with proper features'
            })
        
        # Issue 5: Poor model performance indicators
        best_r2 = results['best_test_r2']
        if best_r2 < 0:
            issues.append({
                'severity': 'CRITICAL',
                'issue': 'Models Worse Than Baseline',
                'description': f'Best R² = {best_r2:.3f} (negative)',
                'impact': 'Models perform worse than predicting the mean',
                'solution': 'Fundamental data or methodology problems need fixing'
            })
        
        return issues
    
    def generate_recommendations(self, issues, data_analysis):
        """Generate specific recommendations to fix the problems"""
        logger.info("Generating improvement recommendations...")
        
        recommendations = {
            'immediate_fixes': [],
            'data_collection': [],
            'feature_engineering': [],
            'modeling_strategy': []
        }
        
        # Immediate fixes
        if any(issue['issue'] == 'Severe Class Imbalance' for issue in issues):
            recommendations['immediate_fixes'].extend([
                'Implement stratified sampling to balance system types',
                'Use class weights in models to account for imbalance',
                'Consider separate models for each system type'
            ])
        
        if any(issue['issue'] == 'Extreme Target Range' for issue in issues):
            recommendations['immediate_fixes'].extend([
                'Apply log transformation to power density target',
                'Use robust scaling for extreme outliers',
                'Consider separate models for different power ranges'
            ])
        
        # Data collection priorities
        battery_percentage = data_analysis['system_imbalance']['percentages'].get('NASA_battery', 0)
        if battery_percentage > 50:
            recommendations['data_collection'].extend([
                'Priority 1: Find large fuel cell datasets (NREL, DOE)',
                'Priority 2: Collect more MFC experimental data',
                'Priority 3: Search for supercapacitor datasets',
                'Target: Balance dataset to 40% battery, 30% fuel cell, 30% other'
            ])
        
        recommendations['data_collection'].extend([
            'Search Battery500 consortium for high-quality battery data',
            'Contact university research groups for MFC data sharing',
            'Mine recent electrochemical papers for tabulated data',
            'Target: Increase from 2,619 to 10,000+ real samples'
        ])
        
        # Feature engineering improvements
        if any(issue['issue'] == 'No Materials Information' for issue in issues):
            recommendations['feature_engineering'].extend([
                'Add electrode material properties (conductivity, surface area)',
                'Include electrolyte composition and ionic conductivity',
                'Add catalyst loading and activity descriptors',
                'Include system design parameters (electrode spacing, flow rates)'
            ])
        
        recommendations['feature_engineering'].extend([
            'Replace binary system flags with continuous performance metrics',
            'Add physics-based features (overpotential, mass transport)',
            'Create proper interaction terms for material-condition combinations',
            'Remove redundant temperature variants (keep only temperature_c)'
        ])
        
        # Modeling strategy improvements
        recommendations['modeling_strategy'].extend([
            'Implement domain-specific models (separate for MFC, fuel cell, battery)',
            'Use ensemble methods to handle data heterogeneity',
            'Add uncertainty quantification for prediction confidence',
            'Implement proper cross-validation by data source or time'
        ])
        
        return recommendations
    
    def create_diagnostic_report(self):
        """Create comprehensive diagnostic report"""
        logger.info("Creating comprehensive diagnostic report...")
        
        # Load data and results
        df, results = self.load_data_and_results()
        
        # Run analyses
        data_analysis = self.analyze_data_distribution(df)
        feature_problems = self.analyze_feature_problems(df)
        core_issues = self.identify_core_issues(df, results, data_analysis)
        recommendations = self.generate_recommendations(core_issues, data_analysis)
        
        # Create report
        report = {
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'dataset_summary': {
                'total_samples': len(df),
                'features': len([col for col in df.columns if col not in ['power_density', 'data_source']]),
                'system_types': len(df['data_source'].unique())
            },
            'data_distribution_analysis': data_analysis,
            'feature_problems': feature_problems,
            'core_failure_issues': core_issues,
            'improvement_recommendations': recommendations,
            'current_performance': {
                'best_model': results['best_model'],
                'best_r2': results['best_test_r2'],
                'assessment': 'FAILING - Requires fundamental fixes'
            }
        }
        
        return report
    
    def save_diagnostic_report(self, report):
        """Save diagnostic report"""
        output_dir = Path("analysis")
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / "model_failure_diagnostic.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Create human-readable summary
        self.create_readable_summary(report, output_dir)
        
        return output_dir / "model_failure_diagnostic.json"
    
    def create_readable_summary(self, report, output_dir):
        """Create human-readable diagnostic summary"""
        
        summary = f"""
# Model Failure Diagnostic Report

## Current Status: FAILING ❌
**Best Model**: {report['current_performance']['best_model']}
**Best R²**: {report['current_performance']['best_r2']:.4f}
**Assessment**: Models perform worse than predicting the mean

## Critical Issues Identified

"""
        
        # Add critical issues
        critical_issues = [issue for issue in report['core_failure_issues'] 
                          if issue['severity'] == 'CRITICAL']
        
        for i, issue in enumerate(critical_issues, 1):
            summary += f"""
### {i}. {issue['issue']} 🚨
**Problem**: {issue['description']}
**Impact**: {issue['impact']}
**Solution**: {issue['solution']}
"""
        
        # Add data problems
        summary += f"""
## Data Quality Problems

### Data Imbalance
- **Imbalance Ratio**: {report['data_distribution_analysis']['system_imbalance']['imbalance_ratio']:.1f}:1
- **Dominant System**: {report['data_distribution_analysis']['system_imbalance']['dominant_system'][0]} ({report['data_distribution_analysis']['system_imbalance']['dominant_system'][1]:.1f}%)

### Target Variable Issues
- **Power Range**: {report['data_distribution_analysis']['power_distribution']['range_ratio']:.0f}:1 ratio
- **Distribution**: Highly skewed (skewness = {report['data_distribution_analysis']['power_distribution']['skewness']:.2f})

## Immediate Action Plan

### Phase 1: Data Collection (Week 1-2)
"""
        
        for rec in report['improvement_recommendations']['data_collection'][:4]:
            summary += f"- {rec}\n"
        
        summary += """
### Phase 2: Feature Engineering (Week 3)
"""
        
        for rec in report['improvement_recommendations']['feature_engineering'][:3]:
            summary += f"- {rec}\n"
        
        summary += """
### Phase 3: Modeling Strategy (Week 4)
"""
        
        for rec in report['improvement_recommendations']['modeling_strategy'][:3]:
            summary += f"- {rec}\n"
        
        summary += f"""
## Success Metrics
- **Target R²**: 0.7+ for each system type
- **Data Balance**: <5:1 imbalance ratio between system types
- **Sample Size**: 10,000+ real experimental samples
- **Feature Count**: 20+ physics-based features

## Next Steps
1. Implement immediate fixes from Phase 1
2. Search for datasets listed in ESSENTIAL_DATASETS_TO_FIND.md
3. Focus on fuel cell and MFC data to balance the dataset
4. Add materials properties and operating conditions

*Report generated: {report['analysis_timestamp']}*
"""
        
        with open(output_dir / "diagnostic_summary.md", 'w') as f:
            f.write(summary)


if __name__ == "__main__":
    analyzer = ModelFailureAnalyzer()
    report = analyzer.create_diagnostic_report()
    report_file = analyzer.save_diagnostic_report(report)
    
    print(f"\n🔍 Model Failure Analysis Complete")
    print(f"📁 Report saved to: {report_file}")
    print(f"\n🚨 Critical Issues Found: {len([i for i in report['core_failure_issues'] if i['severity'] == 'CRITICAL'])}")
    print(f"💡 Recommendations Generated: {len(report['improvement_recommendations']['immediate_fixes']) + len(report['improvement_recommendations']['data_collection'])}")
    print(f"\n👀 See analysis/diagnostic_summary.md for action plan")
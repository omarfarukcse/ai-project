# scripts/security_audit.py
import subprocess
import json
import sys
from typing import Dict, List, Any
import datetime

from src.logger import get_logger

logger = get_logger(__name__)

class SecurityAuditor:
    """Automated dependency security auditing"""
    
    def __init__(self, fail_on_critical: bool = True):
        self.fail_on_critical = fail_on_critical
        self.results = {
            'timestamp': datetime.datetime.now().isoformat(),
            'vulnerabilities': [],
            'summary': {}
        }
    
    def run_pip_audit(self) -> Dict:
        """Run pip-audit to check for vulnerabilities"""
        try:
            result = subprocess.run(
                ['pip-audit', '--format', 'json', '--desc'],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.stdout:
                audit_data = json.loads(result.stdout)
                return self._parse_pip_audit(audit_data)
            
            return {'vulnerabilities': [], 'count': 0}
            
        except Exception as e:
            logger.error(f"pip-audit failed: {e}")
            return {'error': str(e), 'vulnerabilities': [], 'count': 0}
    
    def run_safety_check(self) -> Dict:
        """Run safety check for vulnerabilities"""
        try:
            result = subprocess.run(
                ['safety', 'check', '--json'],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.stdout:
                safety_data = json.loads(result.stdout)
                return self._parse_safety_check(safety_data)
            
            return {'vulnerabilities': [], 'count': 0}
            
        except Exception as e:
            logger.error(f"Safety check failed: {e}")
            return {'error': str(e), 'vulnerabilities': [], 'count': 0}
    
    def _parse_pip_audit(self, data: Dict) -> Dict:
        """Parse pip-audit output"""
        vulnerabilities = []
        
        for item in data.get('vulnerabilities', []):
            vuln = {
                'package': item.get('name'),
                'version': item.get('version'),
                'vulnerability_id': item.get('id'),
                'severity': item.get('severity', 'UNKNOWN'),
                'description': item.get('description', ''),
                'fix_version': item.get('fix_version')
            }
            vulnerabilities.append(vuln)
        
        return {
            'vulnerabilities': vulnerabilities,
            'count': len(vulnerabilities),
            'critical_count': sum(1 for v in vulnerabilities if v['severity'] in ['CRITICAL', 'HIGH'])
        }
    
    def _parse_safety_check(self, data: Dict) -> Dict:
        """Parse safety check output"""
        vulnerabilities = []
        
        for item in data.get('vulnerabilities', []):
            vuln = {
                'package': item.get('package_name'),
                'version': item.get('installed_version'),
                'vulnerability_id': item.get('vulnerability_id'),
                'severity': item.get('severity', 'UNKNOWN'),
                'description': item.get('description', ''),
                'fix_version': item.get('fixed_version')
            }
            vulnerabilities.append(vuln)
        
        return {
            'vulnerabilities': vulnerabilities,
            'count': len(vulnerabilities),
            'critical_count': sum(1 for v in vulnerabilities if v['severity'] in ['CRITICAL', 'HIGH'])
        }
    
    def audit(self) -> bool:
        """Run full security audit"""
        print("🔒 Running dependency security audit...")
        
        # Run pip-audit
        pip_results = self.run_pip_audit()
        self.results['vulnerabilities'].extend(pip_results.get('vulnerabilities', []))
        self.results['summary']['pip_audit'] = pip_results
        
        # Run safety check
        safety_results = self.run_safety_check()
        self.results['vulnerabilities'].extend(safety_results.get('vulnerabilities', []))
        self.results['summary']['safety_check'] = safety_results
        
        # Generate report
        self._generate_report()
        
        # Determine if audit passes
        critical_vulns = sum(1 for v in self.results['vulnerabilities'] 
                           if v.get('severity') in ['CRITICAL', 'HIGH'])
        
        if critical_vulns > 0:
            print(f"❌ Found {critical_vulns} critical/high vulnerabilities!")
            
            if self.fail_on_critical:
                print("❌ Audit failed - critical vulnerabilities found")
                return False
        
        print(f"✅ Audit passed - {len(self.results['vulnerabilities'])} vulnerabilities found")
        return True
    
    def _generate_report(self):
        """Generate security audit report"""
        report_file = "outputs/security_audit_report.json"
        
        # Add recommendations
        self.results['recommendations'] = []
        for vuln in self.results['vulnerabilities']:
            if vuln.get('fix_version'):
                self.results['recommendations'].append(
                    f"Update {vuln['package']} to {vuln['fix_version']} to fix {vuln['vulnerability_id']}"
                )
        
        # Save report
        import json
        from pathlib import Path
        
        Path("outputs").mkdir(exist_ok=True)
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"📊 Report saved to {report_file}")

def main():
    auditor = SecurityAuditor(fail_on_critical=True)
    success = auditor.audit()
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
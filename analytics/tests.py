import io
from django.test import TestCase, Client
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
from .models import PatientRecord
from .services import predict_single_patient, process_batch_csv, get_biomarker_status

class MLServicesTestCase(TestCase):
    def test_single_prediction_low_risk(self):
        data = {'glucose': 85.0, 'bmi': 21.5, 'age': 28, 'bp': 70.0, 'insulin': 40.0}
        result = predict_single_patient(data)
        self.assertFalse(result['is_high_risk'])
        self.assertLess(result['risk_percentage'], 35.0)
        self.assertEqual(result['tier_color'], 'success')
        self.assertEqual(len(result['biomarkers']), 5)
        self.assertGreater(len(result['recommendations']), 0)

    def test_single_prediction_high_risk(self):
        data = {'glucose': 180.0, 'bmi': 36.0, 'age': 62, 'bp': 98.0, 'insulin': 220.0}
        result = predict_single_patient(data)
        self.assertTrue(result['is_high_risk'])
        self.assertGreaterEqual(result['risk_percentage'], 50.0)
        self.assertEqual(result['risk_tier'], 'High Risk')
        self.assertEqual(result['tier_color'], 'danger')

    def test_biomarker_status_ranges(self):
        status = get_biomarker_status('glucose', 140)
        self.assertEqual(status['flag'], 'high')
        status_bmi = get_biomarker_status('bmi', 22.0)
        self.assertEqual(status_bmi['flag'], 'normal')

    def test_batch_csv_processing_with_aliases(self):
        csv_text = (
            "Patient Name,Glucose,BMI,Age,Blood Pressure,Insulin\n"
            "Alice,85,22,29,70,45\n"
            "Bob,175,35,60,95,200\n"
        )
        records, summary, export_str = process_batch_csv(io.StringIO(csv_text))
        self.assertEqual(summary['total'], 2)
        self.assertEqual(summary['high_risk'], 1)
        self.assertEqual(summary['low_risk'], 1)
        self.assertEqual(records[0]['name'], 'Alice')
        self.assertIn('Risk_Probability', export_str)

    def test_batch_csv_missing_features(self):
        csv_text = "name,glucose,age\nAlice,90,30\n"
        with self.assertRaises(ValueError):
            process_batch_csv(io.StringIO(csv_text))


class DashboardViewsTestCase(TestCase):
    def setUp(self):
        self.client = Client()

    def test_dashboard_get(self):
        response = self.client.get(reverse('analytics:dashboard'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'HealthMetrics ML')
        self.assertContains(response, 'Model Performance')

    def test_single_patient_post_valid(self):
        payload = {
            'single_predict': '1',
            'name': 'Clara Hayes',
            'glucose': '90.0',
            'bmi': '22.4',
            'age': '34',
            'bp': '72.0',
            'insulin': '48.0'
        }
        response = self.client.post(reverse('analytics:dashboard'), payload)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Clara Hayes')
        self.assertEqual(PatientRecord.objects.count(), 1)
        record = PatientRecord.objects.first()
        self.assertEqual(record.name, 'Clara Hayes')

    def test_single_patient_post_invalid_glucose(self):
        payload = {
            'single_predict': '1',
            'name': 'Test Error',
            'glucose': '9999',
            'bmi': '22.0',
            'age': '30',
            'bp': '70.0',
            'insulin': '40.0'
        }
        response = self.client.post(reverse('analytics:dashboard'), payload)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Fasting Glucose must be between 20 and 600')
        self.assertEqual(PatientRecord.objects.count(), 0)

    def test_download_sample_csv(self):
        response = self.client.get(reverse('analytics:download_sample_csv'))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'text/csv')
        self.assertIn(b'Sarah Jenkins', response.content)

    def test_batch_upload_and_export(self):
        csv_bytes = b"name,glucose,bmi,age,bp,insulin\nSarah,85,21.5,30,68,40\n"
        uploaded_file = SimpleUploadedFile("cohort.csv", csv_bytes, content_type="text/csv")
        post_resp = self.client.post(reverse('analytics:dashboard'), {
            'batch_predict': '1',
            'csv_file': uploaded_file
        })
        self.assertEqual(post_resp.status_code, 200)
        self.assertContains(post_resp, 'Cohort Stratification Results')

        export_resp = self.client.get(reverse('analytics:export_batch_csv'))
        self.assertEqual(export_resp.status_code, 200)
        self.assertIn(b'Risk_Probability', export_resp.content)

    def test_clear_history(self):
        PatientRecord.objects.create(
            name='Test', glucose=100, bmi=24, age=30, bp=75, insulin=50,
            risk_percentage=15.0, status='Normal'
        )
        self.assertEqual(PatientRecord.objects.count(), 1)
        resp = self.client.post(reverse('analytics:clear_history'))
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(PatientRecord.objects.count(), 0)


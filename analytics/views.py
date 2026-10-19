from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.contrib import messages
from .models import PatientRecord
from .services import predict_single_patient, process_batch_csv, model_metrics

def dashboard_view(request):
    result = None
    batch_records = None
    batch_summary = None
    error = None
    success_msg = None
    form_data = {
        'name': '',
        'glucose': '',
        'bmi': '',
        'age': '',
        'bp': '',
        'insulin': '',
    }

    # Handle single patient submission
    if request.method == 'POST' and 'single_predict' in request.POST:
        raw_name = request.POST.get('name', '').strip()
        raw_glucose = request.POST.get('glucose', '').strip()
        raw_bmi = request.POST.get('bmi', '').strip()
        raw_age = request.POST.get('age', '').strip()
        raw_bp = request.POST.get('bp', '').strip()
        raw_insulin = request.POST.get('insulin', '').strip()

        form_data = {
            'name': raw_name or 'Anonymous Patient',
            'glucose': raw_glucose,
            'bmi': raw_bmi,
            'age': raw_age,
            'bp': raw_bp,
            'insulin': raw_insulin,
        }

        # Validation
        try:
            if not raw_glucose or not raw_bmi or not raw_age or not raw_bp or not raw_insulin:
                raise ValueError("All biometric input fields are required.")

            glucose = float(raw_glucose)
            bmi = float(raw_bmi)
            age = int(raw_age)
            bp = float(raw_bp)
            insulin = float(raw_insulin)

            if not (20 <= glucose <= 600):
                raise ValueError("Fasting Glucose must be between 20 and 600 mg/dL.")
            if not (10 <= bmi <= 80):
                raise ValueError("BMI must be between 10.0 and 80.0 kg/m².")
            if not (1 <= age <= 125):
                raise ValueError("Patient Age must be between 1 and 125 years.")
            if not (30 <= bp <= 250):
                raise ValueError("Blood Pressure must be between 30 and 250 mm Hg.")
            if not (0 <= insulin <= 1000):
                raise ValueError("Insulin must be between 0 and 1000 µIU/mL.")

            patient_data = {
                'glucose': glucose,
                'bmi': bmi,
                'age': age,
                'bp': bp,
                'insulin': insulin,
            }
            name = raw_name if raw_name else 'Anonymous Patient'

            # Run ML decision support inference
            result = predict_single_patient(patient_data)
            result['patient_name'] = name
            result['input_data'] = patient_data

            # Persist evaluation to live database
            record = PatientRecord.objects.create(
                name=name,
                glucose=patient_data['glucose'],
                bmi=patient_data['bmi'],
                age=patient_data['age'],
                bp=patient_data['bp'],
                insulin=patient_data['insulin'],
                risk_percentage=result['risk_percentage'],
                status=result['status']
            )
            result['record_id'] = record.id
            success_msg = f"Evaluation completed and logged for {name}."

        except ValueError as ve:
            error = str(ve)
        except Exception as e:
            error = f"Inference engine error: {str(e)}"

    # Handle batch CSV upload
    elif request.method == 'POST' and 'batch_predict' in request.POST:
        if 'csv_file' in request.FILES:
            try:
                uploaded_file = request.FILES['csv_file']
                if not uploaded_file.name.lower().endswith('.csv'):
                    raise ValueError("Please upload a file with a .csv extension.")

                records, summary, export_str = process_batch_csv(uploaded_file)
                batch_records = records[:25]  # Preview top 25 records
                batch_summary = summary
                
                # Cache CSV string in session for instant export download
                if hasattr(request, 'session'):
                    request.session['batch_csv_export'] = export_str
                    request.session['batch_filename'] = f"scored_{uploaded_file.name}"
                success_msg = f"Successfully processed {summary['total']} cohort records."
            except Exception as e:
                error = f"Batch scoring failed: {str(e)}"
        else:
            error = "No CSV file was selected for upload."

    # Live historical queries
    recent_patients = PatientRecord.objects.all()[:10]
    total_records = PatientRecord.objects.count()
    high_risk_records = PatientRecord.objects.filter(risk_percentage__gte=50.0).count()
    low_risk_records = PatientRecord.objects.filter(risk_percentage__lt=30.0).count()
    has_batch_export = bool(request.session.get('batch_csv_export')) if hasattr(request, 'session') else False

    # Compute average risk from history
    from django.db.models import Avg
    avg_risk_history = PatientRecord.objects.aggregate(avg=Avg('risk_percentage'))['avg']
    avg_risk_history = round(avg_risk_history, 1) if avg_risk_history else 0

    context = {
        'result': result,
        'batch_records': batch_records,
        'batch_summary': batch_summary,
        'has_batch_export': has_batch_export,
        'recent_patients': recent_patients,
        'total_records': total_records,
        'high_risk_records': high_risk_records,
        'low_risk_records': low_risk_records,
        'avg_risk_history': avg_risk_history,
        'metrics': model_metrics,
        'form_data': form_data,
        'error': error,
        'success_msg': success_msg
    }
    return render(request, 'analytics/dashboard.html', context)


def download_sample_csv(request):
    """Generates and serves a realistic sample CSV template for testing batch cohort scoring."""
    sample_content = (
        "name,glucose,bmi,age,bp,insulin\n"
        "Sarah Jenkins,88.5,22.1,29,68.0,42.0\n"
        "Marcus Sterling,156.0,33.4,54,92.0,175.0\n"
        "Elena Rostova,108.0,26.8,44,82.0,95.0\n"
        "David Chen,94.0,24.0,36,74.0,58.0\n"
        "Amara Okafor,178.5,35.2,61,96.0,210.0\n"
        "Liam Gallagher,124.0,28.5,50,86.0,118.0\n"
        "Sophia Martinez,91.0,21.8,31,70.0,46.0\n"
        "Klaus Schmidt,142.0,30.5,58,88.0,140.0\n"
        "Aria Patel,112.0,27.2,46,80.0,88.0\n"
        "James Wilson,189.0,37.1,65,102.0,245.0\n"
    )
    response = HttpResponse(sample_content, content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="healthmetrics_sample_cohort.csv"'
    return response


def export_batch_csv(request):
    """Downloads the scored batch dataset with risk probability and clinical tiers."""
    csv_data = request.session.get('batch_csv_export') if hasattr(request, 'session') else None
    filename = request.session.get('batch_filename', 'scored_healthmetrics_cohort.csv') if hasattr(request, 'session') else 'scored_healthmetrics_cohort.csv'
    if not csv_data:
        return redirect('analytics:dashboard')
    
    response = HttpResponse(csv_data, content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    return response



def clear_history(request):
    """Clears all patient records from the database."""
    if request.method == 'POST':
        PatientRecord.objects.all().delete()
    return redirect('analytics:dashboard')
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from django.conf import settings
from django.core.files.storage import default_storage
from django.http import FileResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST

from .ml_model import predict_image


# ============================================================
# IMAGE UPLOAD SETTINGS
# ============================================================

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


# ============================================================
# NORMAL IMAGE UPLOAD + PREDICTION
# ============================================================

def home(request):

    context = {
        "result": None,
        "confidence": None,
        "image_url": None,
        "error": None,
    }

    if request.method == "POST":

        uploaded_file = request.FILES.get("image")

        # ----------------------------------------------------
        # Check file
        # ----------------------------------------------------

        if not uploaded_file:

            context["error"] = "Please select an image."

            return render(
                request,
                "detector/home.html",
                context
            )

        # ----------------------------------------------------
        # Check extension
        # ----------------------------------------------------

        extension = Path(
            uploaded_file.name
        ).suffix.lower()

        if extension not in ALLOWED_EXTENSIONS:

            context["error"] = (
                "Invalid image format. "
                "Please upload JPG, JPEG, PNG or BMP."
            )

            return render(
                request,
                "detector/home.html",
                context
            )

        # ----------------------------------------------------
        # Check file size
        # ----------------------------------------------------

        if uploaded_file.size > MAX_FILE_SIZE:

            context["error"] = (
                "Image size must be less than 10 MB."
            )

            return render(
                request,
                "detector/home.html",
                context
            )

        # ----------------------------------------------------
        # Save uploaded image
        # ----------------------------------------------------

        filename = (
            f"{uuid4().hex}{extension}"
        )

        relative_path = (
            f"uploads/{filename}"
        )

        saved_path = default_storage.save(
            relative_path,
            uploaded_file
        )

        full_path = default_storage.path(
            saved_path
        )

        # ----------------------------------------------------
        # CNN Prediction
        # ----------------------------------------------------

        try:

            prediction = predict_image(
                full_path
            )

            context["result"] = (
                prediction["result"]
            )

            context["confidence"] = (
                prediction["confidence"]
            )

            context["image_url"] = (
                default_storage.url(
                    saved_path
                )
            )

        except Exception as error:

            context["error"] = (
                f"Prediction failed: {error}"
            )

    return render(
        request,
        "detector/home.html",
        context
    )


# ============================================================
# LIVE DETECTION PAGE
# ============================================================

def live_detection(request):

    return render(
        request,
        "detector/live_detection.html"
    )


# ============================================================
# SESSION ID VALIDATION
# ============================================================

def get_safe_session_id(session_id):

    if not session_id:
        return None

    if not re.fullmatch(
        r"session_[A-Za-z0-9]+",
        session_id
    ):
        return None

    return session_id


# ============================================================
# LIVE CNN DETECTION
# ============================================================

@require_POST
def live_detect(request):

    uploaded_file = request.FILES.get(
        "image"
    )

    session_id = request.POST.get(
        "session_id"
    )

    session_id = get_safe_session_id(
        session_id
    )

    # --------------------------------------------------------
    # Validate frame
    # --------------------------------------------------------

    if not uploaded_file:

        return JsonResponse({
            "success": False,
            "error": "No frame received."
        })

    # --------------------------------------------------------
    # Validate session
    # --------------------------------------------------------

    if not session_id:

        return JsonResponse({
            "success": False,
            "error": "Invalid session."
        })

    # --------------------------------------------------------
    # Create session report folder
    # --------------------------------------------------------

    session_folder = (
        settings.LIVE_REPORTS_ROOT
        / session_id
    )

    session_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Temporarily save webcam frame
    # --------------------------------------------------------

    frame_path = (
        session_folder
        / "current_frame.jpg"
    )

    with open(
        frame_path,
        "wb"
    ) as destination:

        for chunk in uploaded_file.chunks():

            destination.write(chunk)

    try:

        # ----------------------------------------------------
        # CNN Prediction
        # ----------------------------------------------------

        prediction = predict_image(
            str(frame_path)
        )

        result = prediction["result"]

        confidence = prediction["confidence"]

        # ----------------------------------------------------
        # Save detection record
        # ----------------------------------------------------

        csv_path = (
            session_folder
            / "detections.csv"
        )

        file_exists = csv_path.exists()

        with open(
            csv_path,
            "a",
            newline="",
            encoding="utf-8"
        ) as csv_file:

            writer = csv.writer(
                csv_file
            )

            if not file_exists:

                writer.writerow([
                    "timestamp",
                    "result",
                    "confidence"
                ])

            writer.writerow([
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                result,
                confidence
            ])

        # ----------------------------------------------------
        # Return prediction to browser
        # ----------------------------------------------------

        return JsonResponse({
            "success": True,
            "result": result,
            "confidence": confidence
        })

    except Exception as error:

        return JsonResponse({
            "success": False,
            "error": str(error)
        })

    finally:

        # ----------------------------------------------------
        # Delete temporary frame
        # ----------------------------------------------------

        try:
            if frame_path.exists():
                frame_path.unlink()
        except Exception:
            pass


# ============================================================
# GENERATE REPORT
# ============================================================

@require_POST
def generate_report(request):

    session_id = request.POST.get(
        "session_id"
    )

    session_id = get_safe_session_id(
        session_id
    )

    # --------------------------------------------------------
    # Validate session
    # --------------------------------------------------------

    if not session_id:

        return JsonResponse({
            "success": False,
            "error": "Invalid session."
        })

    # --------------------------------------------------------
    # Session folder
    # --------------------------------------------------------

    session_folder = (
        settings.LIVE_REPORTS_ROOT
        / session_id
    )

    csv_path = (
        session_folder
        / "detections.csv"
    )

    # --------------------------------------------------------
    # Check detection records
    # --------------------------------------------------------

    if not csv_path.exists():

        return JsonResponse({
            "success": False,
            "error": "No detection records found."
        })

    # --------------------------------------------------------
    # Read detection records
    # --------------------------------------------------------

    detections = []

    with open(
        csv_path,
        "r",
        encoding="utf-8"
    ) as csv_file:

        reader = csv.DictReader(
            csv_file
        )

        for row in reader:

            detections.append(row)

    # --------------------------------------------------------
    # Calculate statistics
    # --------------------------------------------------------

    total = len(detections)

    crack_count = sum(
        1
        for item in detections
        if item["result"] == "Crack Detected"
    )

    no_crack_count = sum(
        1
        for item in detections
        if item["result"] == "No Crack"
    )

    confidence_values = []

    for item in detections:

        try:
            confidence_values.append(
                float(item["confidence"])
            )
        except (ValueError, TypeError):
            pass

    average_confidence = 0

    if confidence_values:

        average_confidence = (
            sum(confidence_values)
            / len(confidence_values)
        )

    crack_percentage = 0

    no_crack_percentage = 0

    if total > 0:

        crack_percentage = (
            crack_count / total
        ) * 100

        no_crack_percentage = (
            no_crack_count / total
        ) * 100

    # --------------------------------------------------------
    # Report data
    # --------------------------------------------------------

    report = {

        "project":
            "Industrial Surface Crack Detection",

        "model":
            "Baseline V1 CNN",

        "session_id":
            session_id,

        "generated_at":
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

        "total_detections":
            total,

        "crack_detections":
            crack_count,

        "no_crack_detections":
            no_crack_count,

        "crack_percentage":
            round(
                crack_percentage,
                2
            ),

        "no_crack_percentage":
            round(
                no_crack_percentage,
                2
            ),

        "average_confidence":
            round(
                average_confidence,
                2
            ),
    }

    # ========================================================
    # SAVE JSON REPORT
    # ========================================================

    json_path = (
        session_folder
        / "report.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as json_file:

        json.dump(
            report,
            json_file,
            indent=4
        )

    # ========================================================
    # CREATE DETECTION TABLE
    # ========================================================

    rows_html = ""

    for item in detections:

        if item["result"] == "Crack Detected":

            result_class = "crack"

        else:

            result_class = "safe"

        rows_html += f"""
        <tr>
            <td>{item["timestamp"]}</td>

            <td class="{result_class}">
                {item["result"]}
            </td>

            <td>
                {item["confidence"]}%
            </td>
        </tr>
        """

    # ========================================================
    # CREATE HUMAN-READABLE HTML REPORT
    # ========================================================

    html_path = (
        session_folder
        / "report.html"
    )

    html_content = f"""
<!DOCTYPE html>

<html>

<head>

    <meta charset="UTF-8">

    <title>
        Industrial Surface Crack Detection Report
    </title>

    <style>

        body {{
            font-family: Arial, sans-serif;
            background: #eef2f7;
            padding: 40px;
            color: #1f2937;
        }}

        .report {{
            max-width: 900px;
            margin: auto;
            background: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow:
                0 10px 30px
                rgba(0, 0, 0, 0.10);
        }}

        h1 {{
            text-align: center;
            color: #172554;
            margin-bottom: 30px;
        }}

        h2 {{
            color: #1e3a8a;
            margin-top: 30px;
        }}

        .info {{
            background: #f8fafc;
            padding: 20px;
            border-radius: 10px;
            line-height: 1.8;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}

        th,
        td {{
            padding: 14px;
            border-bottom: 1px solid #ddd;
            text-align: left;
        }}

        th {{
            background: #f1f5f9;
        }}

        .crack {{
            color: #dc2626;
            font-weight: bold;
        }}

        .safe {{
            color: #16a34a;
            font-weight: bold;
        }}

        .summary {{
            display: grid;
            grid-template-columns:
                repeat(3, 1fr);
            gap: 15px;
            margin-top: 25px;
        }}

        .box {{
            background: #f8fafc;
            padding: 20px;
            text-align: center;
            border-radius: 10px;
        }}

        .box strong {{
            display: block;
            font-size: 28px;
            color: #1e3a8a;
            margin-top: 8px;
        }}

        .footer {{
            margin-top: 30px;
            text-align: center;
            color: #64748b;
            font-size: 14px;
        }}

    </style>

</head>

<body>

<div class="report">

    <h1>
        Industrial Surface Crack Detection
    </h1>

    <div class="info">

        <p>
            <strong>Model:</strong>
            Baseline V1 CNN
        </p>

        <p>
            <strong>Session:</strong>
            {session_id}
        </p>

        <p>
            <strong>Generated:</strong>
            {report["generated_at"]}
        </p>

    </div>


    <h2>
        Detection Summary
    </h2>


    <div class="summary">

        <div class="box">

            Total Detections

            <strong>
                {total}
            </strong>

        </div>


        <div class="box">

            Cracks Detected

            <strong class="crack">
                {crack_count}
            </strong>

        </div>


        <div class="box">

            No Crack

            <strong class="safe">
                {no_crack_count}
            </strong>

        </div>

    </div>


    <h2>
        Statistics
    </h2>


    <table>

        <tr>

            <th>
                Metric
            </th>

            <th>
                Value
            </th>

        </tr>


        <tr>

            <td>
                Crack Percentage
            </td>

            <td>
                {crack_percentage:.2f}%
            </td>

        </tr>


        <tr>

            <td>
                No Crack Percentage
            </td>

            <td>
                {no_crack_percentage:.2f}%
            </td>

        </tr>


        <tr>

            <td>
                Average Confidence
            </td>

            <td>
                {average_confidence:.2f}%
            </td>

        </tr>

    </table>


    <h2>
        Detection Records
    </h2>


    <table>

        <tr>

            <th>
                Time
            </th>

            <th>
                Result
            </th>

            <th>
                Confidence
            </th>

        </tr>

        {rows_html}

    </table>


    <div class="footer">

        Industrial Surface Crack Detection
        |
        Django + CNN

    </div>

</div>

</body>

</html>
"""

    # --------------------------------------------------------
    # Save HTML report
    # --------------------------------------------------------

    with open(
        html_path,
        "w",
        encoding="utf-8"
    ) as html_file:

        html_file.write(
            html_content
        )

    # --------------------------------------------------------
    # Return report URL
    # --------------------------------------------------------

    return JsonResponse({

        "success": True,

        "report_url":
            f"/report/{session_id}/",

    })


# ============================================================
# VIEW GENERATED REPORT
# ============================================================

def view_report(
    request,
    session_id
):

    session_id = get_safe_session_id(
        session_id
    )

    # --------------------------------------------------------
    # Validate session
    # --------------------------------------------------------

    if not session_id:

        return JsonResponse(
            {
                "error": "Invalid session."
            },
            status=400
        )

    # --------------------------------------------------------
    # Find report
    # --------------------------------------------------------

    report_path = (
        settings.LIVE_REPORTS_ROOT
        / session_id
        / "report.html"
    )

    # --------------------------------------------------------
    # Check report
    # --------------------------------------------------------

    if not report_path.exists():

        return JsonResponse(
            {
                "error": "Report not found."
            },
            status=404
        )

    # --------------------------------------------------------
    # Return HTML report
    # --------------------------------------------------------

    return FileResponse(
        open(
            report_path,
            "rb"
        ),
        content_type="text/html"
    )
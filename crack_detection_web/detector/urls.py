from django.urls import path

from . import views


urlpatterns = [

    # --------------------------------------------------------
    # Home / Image Upload
    # --------------------------------------------------------

    path(
        "",
        views.home,
        name="home"
    ),

    # --------------------------------------------------------
    # Live Detection Page
    # --------------------------------------------------------

    path(
        "live/",
        views.live_detection,
        name="live_detection"
    ),

    # --------------------------------------------------------
    # Live CNN Detection API
    # --------------------------------------------------------

    path(
        "live-detect/",
        views.live_detect,
        name="live_detect"
    ),

    # --------------------------------------------------------
    # Generate Detection Report
    # --------------------------------------------------------

    path(
        "generate-report/",
        views.generate_report,
        name="generate_report"
    ),

    # --------------------------------------------------------
    # View Generated Report
    # --------------------------------------------------------

    path(
        "report/<str:session_id>/",
        views.view_report,
        name="view_report"
    ),
]
# Start the dashboard with the light theme: python run.py
import sys
from streamlit.web import cli

sys.argv = [
    "streamlit", "run", "app.py",
    "--theme.base", "light",
    "--theme.primaryColor", "#E10600",
    "--theme.backgroundColor", "#FFFFFF",
    "--theme.secondaryBackgroundColor", "#F4F5F7",
    "--theme.textColor", "#15151E",
] + sys.argv[1:]   # any extra options you pass, e.g. --server.port 8502
sys.exit(cli.main())

@echo off
call C:\Users\kayra\anaconda3\Scripts\activate.bat coffee_forecast
cd /d C:\Users\kayra\Documents\GitHub\coffee_forecast
python -m streamlit run app.py
pause
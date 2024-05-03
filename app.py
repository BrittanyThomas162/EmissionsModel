from flask import Flask, jsonify, render_template, request, flash, url_for, redirect
from forms import FuelPredictionForm, EmissionForm
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import json
import requests
import xml.etree.ElementTree as ET
from utils import get_country_name_by_code


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'

# Load your model and scaler
fuel_Pred_Model = pickle.load(open("fuel_Pred_Model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb")) 

# Load the LabelEncoder for categorical columns
label_encoder = LabelEncoder()


@app.route('/', methods=['GET', 'POST'])
def fuelPrediction():
    form = FuelPredictionForm()
    if form.validate_on_submit():
        # Collect features from the form
        features_input = [
            form.airline_iata.data.upper(),
            form.acft_icao.data.upper(),
            form.acft_class.data.upper(),
            form.seymour_proxy.data.upper(),
            float(form.seats.data),
            float(form.n_flights.data),
            form.iata_departure.data.upper(),
            form.iata_arrival.data.upper(),
            float(form.distance_km.data),
            float(form.rpk.data),
            float(form.fuel_burn_seymour.data),
            float(form.fuel_burn.data)
        ]

        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame([features_input], columns=['airline_iata', 'acft_icao', 'acft_class', 'seymour_proxy', 'seats', 'n_flights', 'iata_departure', 'iata_arrival', 'distance_km', 'rpk', 'fuel_burn_seymour', 'fuel_burn'])

        # Handle missing values (if any)
        # Assuming you've handled missing values in your training data, you might want to do the same here
        # For example, fill missing values with the mean or median
        # test_df.fillna(test_df.mean(), inplace=True)

        # Encode categorical columns
        categorical_columns = ['airline_iata', 'acft_icao', 'acft_class', 'seymour_proxy', 'iata_departure', 'iata_arrival']
        for column in categorical_columns:
            df[column] = label_encoder.fit_transform(df[column])

        # Convert to numpy array and scale
        features = df.values
        features = scaler.transform(features)

        # Make prediction
        prediction = fuel_Pred_Model.predict(features)

        return render_template('results.html', prediction=prediction[0])
    return render_template('fuelPrediction.html', form=form)


@app.route('/emissions/report', methods=['GET', 'POST'])
def generateReport():
    form = EmissionForm()
    if form.validate_on_submit():
        country_name = get_country_name_by_code(form.country.data)
        raw_data = get_emissions(
            country=form.country.data,
            timeframe=form.timeframe.data,
            start_year=form.start_year.data,
            month=form.month.data if form.month.data else None,
            quarter=form.quarter.data if form.quarter.data else None,
            end_year=form.end_year.data,
            end_month=form.end_month.data if form.end_month.data else None,
            end_quarter=form.end_quarter.data if form.end_quarter.data else None
        )

        if raw_data == "NoRecordsFound":
            flash('No data available for the selected parameters.', 'warning')
            return redirect(url_for('generateReport'))
        elif raw_data == "ErrorParsingXML" or raw_data == "Failed to retrieve data":
            flash('There was an error processing your request. Please try again later.', 'error')
            return redirect(url_for('generateReport'))

        data_summary = {}
        total_emissions = 0
        for item in raw_data:
            time_key = item['time_period']
            emissions = float(item['emissions'])
            total_emissions += emissions
            data_summary.setdefault(time_key, 0)
            data_summary[time_key] += emissions

        return render_template('report.html', data_summary=data_summary, total_emissions=total_emissions, country=country_name, timeframe=form.timeframe.data, start_year=form.start_year.data, end_year=form.end_year.data)
    else:
            # If there is a validation error, the form will be rendered with error messages
            for fieldName, errorMessages in form.errors.items():
                for err in errorMessages:
                    flash(f'Error in {fieldName}: {err}', 'danger')
    return render_template('reportform.html', form=form)


# Works and retruns  a dictionary
def get_emissions(country, timeframe, start_year, month, quarter, end_year, end_month, end_quarter):
    base_url = "https://sdmx.oecd.org/public/rest/data"
    dataflow = "OECD.SDD.NAD.SEEA,DSD_AIR_TRANSPORT@DF_AIR_TRANSPORT,1.0"
    
    # Determine time suffix and period based on the timeframe
    if timeframe == 'annual':
        start_period = f"{start_year}"
        end_period = f"{end_year}"
        time_suffix = ".A......."
    elif timeframe == 'monthly':
        start_period = f"{start_year}-{month.zfill(2)}"
        end_period = f"{end_year}-{end_month.zfill(2)}"
        time_suffix = ".M......."
    elif timeframe == 'quarterly':
        start_period = f"{start_year}-Q{quarter}"
        end_period = f"{end_year}-Q{end_quarter}"
        time_suffix = ".Q......."
    
    url = f"{base_url}/{dataflow}/{country}{time_suffix}?startPeriod={start_period}&endPeriod={end_period}&dimensionAtObservation=AllDimensions"
    print("Requesting URL:", url)  # For debugging purposes
    response = requests.get(url)
    print("response: ", response)
    
    if response.status_code == 404:
        return "NoRecordsFound"
    elif response.status_code == 200:
        try:
            root = ET.fromstring(response.content)
            emissions_data = []
            for obs in root.findall('.//generic:Obs', namespaces={'generic': 'http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic'}):
                data_point = {
                    'time_period': obs.find('.//generic:Value[@id="TIME_PERIOD"]', namespaces={'generic': 'http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic'}).attrib['value'],
                    'emissions': float(obs.find('.//generic:ObsValue', namespaces={'generic': 'http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic'}).attrib['value']),
                    'unit': obs.find('.//generic:Value[@id="UNIT_MEASURE"]', namespaces={'generic': 'http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic'}).attrib['value']
                }
                emissions_data.append(data_point)
            return emissions_data
        except ET.ParseError:
            return "ErrorParsingXML"
    return "Failed to retrieve data"  # Handles other unexpected status codes



if __name__ == '__main__':
    app.run(debug=True)

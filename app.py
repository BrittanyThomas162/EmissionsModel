from flask import Flask, jsonify, render_template, request, flash, url_for, redirect
from forms import FuelPredictionForm, EmissionForm, EmissionRankingForm
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import json
import requests
import xml.etree.ElementTree as ET
# from utils import get_country_name_by_code
from datetime import datetime
from functools import lru_cache


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'

def get_distance(src, dest):
    api_key = 'a741a79a6d7246f8a3e0364dc8'
    urlsrc = f'https://api.checkwx.com/metar/{src}/decoded'
    urldest = f'https://api.checkwx.com/metar/{dest}/decoded'

    responsesrc = requests.get(urlsrc, headers={'X-API-Key': api_key})
    responsedest = requests.get(urldest, headers={'X-API-Key': api_key})

    try:
        srcdata = responsesrc.json()["data"][0]['station']['geometry']['coordinates']
        srclat = srcdata[1]
        srclong = srcdata[0]

        destdata = responsedest.json()["data"][0]['station']['geometry']['coordinates']
        destlat= destdata[1]
        destlong = destdata[0]

        def calc_distance(lat1, lon1, lat2, lon2):
            import math
            # Convert degrees to radians
            lat1_rad = math.radians(lat1)
            lon1_rad = math.radians(lon1)
            lat2_rad = math.radians(lat2)
            lon2_rad = math.radians(lon2)

            # Calculate the difference in longitude
            delta_lon = lon2_rad - lon1_rad

            # Apply the formula
            distance = math.acos(math.sin(lat1_rad) * math.sin(lat2_rad) +
                                math.cos(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon)) * 6371

            return distance

        return calc_distance(srclat, srclong, destlat, destlong)

    except (IndexError, KeyError):
        return None  # Return None or a suitable default if an error occurs


@app.route('/', methods=['GET', 'POST'])
def fuelPrediction():
    # Load your model and scaler
    fuel_Pred_Model = pickle.load(open("fuel_Pred_Model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb")) 
    # Load the LabelEncoder for categorical columns
    label_encoder = LabelEncoder()
    prediction = None  # Initialize prediction and average emissions per flight
    formatted_prediction = None
    average_emissions_per_flight = None
    form = FuelPredictionForm()
    if form.validate_on_submit():
        # Collect features from the form
        airline_iata = form.airline_iata.data.upper()
        acft_icao = form.acft_icao.data.upper()
        seats = float(form.seats.data)
        n_flights = float(form.n_flights.data)
        n = float(form.n_flights.data)
        # iata_departure = form.iata_departure.data.upper()
        # iata_arrival = form.iata_arrival.data.upper()
        icao_departure = form.icao_departure.data.upper()
        icao_arrival = form.icao_arrival.data.upper() 
        fuel_burn_seymour = float(form.fuel_burn_seymour.data)

        # Get the distance between the departure and arrival airports
        distance_km = get_distance(icao_departure, icao_arrival)
        if distance_km is None:
            flash('Failed to calculate distance. Check the ICAO codes.', 'error')
            return render_template('fuelPrediction.html', form=form)
        
        print('calculated distance ', distance_km)
        # Calculate RPK using the seats, distance, and IATA average load factor
        rpk = seats * distance_km * 0.824
        print('calculated rpk ', rpk)

        # Calculate total fuel burn using the number of flights and fuel burn per flight
        fuel_burn = fuel_burn_seymour * n_flights
        print('calculated fuel_burn ', fuel_burn)

        # Prepare data for model
        data = np.array([airline_iata, acft_icao])  # Categorical data
        # data = np.array([airline_iata, acft_icao, iata_departure, iata_arrival])  # Categorical data
        numerical_data = np.array([seats, n_flights, distance_km, rpk,fuel_burn_seymour, fuel_burn])  # Numerical data

         # Encode categorical data
        data_encoded = np.array([label_encoder.fit_transform([feature])[0] for feature in data])

        # Combine and reshape data for scaling
        features = np.concatenate((data_encoded, numerical_data)).reshape(1, -1)
        features_scaled = scaler.transform(features)
        # Make prediction
        prediction = fuel_Pred_Model.predict(features_scaled)
        print('prediction', prediction)
        print('n', n)
        average_emissions_per_flight = prediction[0] / n if n > 0 else 0

        formatted_prediction = "{:,.2f}".format(prediction[0])  # Format the prediction to two decimal places and comma-separated
        average_emissions_per_flight = "{:,.2f}".format(average_emissions_per_flight)
    
    return render_template('fuelPrediction.html', form=form, prediction=formatted_prediction, average_emissions_per_flight=average_emissions_per_flight)

    #     return render_template('results.html', prediction=formatted_prediction, distance=distance_km, rpk=rpk, fuel_burn=fuel_burn)
    # return render_template('fuelPrediction.html', form=form)


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

        if raw_data in ["NoRecordsFound", "ErrorParsingXML", "Failed to retrieve data"]:
            flash('No data available for the selected parameters.' if raw_data == "NoRecordsFound" else 'There was an error processing your request. Please try again later.', 'warning')
            return redirect(url_for('generateReport'))

        sorted_data = sorted(raw_data, key=lambda x: x['time_period'])
        data_summary = {}
        total_emissions = 0
        all_emissions = []

        for item in sorted_data:
            time_key = item['time_period']
            emissions = float(item['emissions'])  # Ensure conversion to float
            total_emissions += emissions
            all_emissions.append(emissions)  # Store raw emissions for min/max calculations
            data_summary[time_key] = emissions

        highest_emissions = max(sorted_data, key=lambda x: x['emissions']) if sorted_data else None
        lowest_emissions = min(sorted_data, key=lambda x: x['emissions']) if sorted_data else None

        average_emissions = total_emissions / len(all_emissions) if all_emissions else 0

        # Format the numbers for display in the template
        formatted_total_emissions = "{:,.2f}".format(total_emissions)
        # formatted_highest_emissions = "{:,.2f}".format(highest_emissions)
        # formatted_lowest_emissions = "{:,.2f}".format(lowest_emissions)
        formatted_average_emissions = "{:,.2f}".format(average_emissions)

        return render_template('report.html', data_summary=data_summary, 
                               total_emissions=formatted_total_emissions,
                               highest_emissions=highest_emissions, 
                               lowest_emissions=lowest_emissions,
                               average_emissions=formatted_average_emissions, 
                               country=country_name, timeframe=form.timeframe.data,
                               start_year=form.start_year.data, end_year=form.end_year.data)
    else:
        for fieldName, errorMessages in form.errors.items():
            for err in errorMessages:
                flash(f'Error in {fieldName}: {err}', 'danger')
    return render_template('reportform.html', form=form)



# @app.route('/emissions/report', methods=['GET', 'POST'])
# def generateReport():
#     form = EmissionForm()
#     if form.validate_on_submit():
#         country_name = get_country_name_by_code(form.country.data)
#         raw_data = get_emissions(
#             country=form.country.data,
#             timeframe=form.timeframe.data,
#             start_year=form.start_year.data,
#             month=form.month.data if form.month.data else None,
#             quarter=form.quarter.data if form.quarter.data else None,
#             end_year=form.end_year.data,
#             end_month=form.end_month.data if form.end_month.data else None,
#             end_quarter=form.end_quarter.data if form.end_quarter.data else None
#         )

#         if raw_data in ["NoRecordsFound", "ErrorParsingXML", "Failed to retrieve data"]:
#             flash('No data available for the selected parameters.' if raw_data == "NoRecordsFound" else 'There was an error processing your request. Please try again later.', 'warning')
#             return redirect(url_for('generateReport'))

#         # Sorting data based on time_period
#         sorted_data = sorted(raw_data, key=lambda x: x['time_period'])

#         # Summarizing data
#         data_summary = {}
#         total_emissions = 0
#         for item in sorted_data:
#             time_key = item['time_period']
#             emissions = item['emissions']
#             total_emissions += emissions
#             data_summary.setdefault(time_key, 0)
#             data_summary[time_key] += emissions
        
#         return render_template('report.html', data_summary=data_summary, total_emissions=total_emissions, country=country_name, timeframe=form.timeframe.data, start_year=form.start_year.data, end_year=form.end_year.data)
#     else:
#         # If there is a validation error, the form will be rendered with error messages
#         for fieldName, errorMessages in form.errors.items():
#             for err in errorMessages:
#                 flash(f'Error in {fieldName}: {err}', 'danger')
#     return render_template('reportform.html', form=form)

# @app.route('/emissions/report', methods=['GET', 'POST'])
# def generateReport():
#     form = EmissionForm()
#     if form.validate_on_submit():
#         country_name = get_country_name_by_code(form.country.data)
#         raw_data = get_emissions(
#             country=form.country.data,
#             timeframe=form.timeframe.data,
#             start_year=form.start_year.data,
#             month=form.month.data if form.month.data else None,
#             quarter=form.quarter.data if form.quarter.data else None,
#             end_year=form.end_year.data,
#             end_month=form.end_month.data if form.end_month.data else None,
#             end_quarter=form.end_quarter.data if form.end_quarter.data else None
#         )

#         if raw_data == "NoRecordsFound":
#             flash('No data available for the selected parameters.', 'warning')
#             return redirect(url_for('generateReport'))
#         elif raw_data == "ErrorParsingXML" or raw_data == "Failed to retrieve data":
#             flash('There was an error processing your request. Please try again later.', 'error')
#             return redirect(url_for('generateReport'))

#         data_summary = {}
#         total_emissions = 0
#         for item in raw_data:
#             time_key = item['time_period']
#             emissions = float(item['emissions'])
#             total_emissions += emissions
#             data_summary.setdefault(time_key, 0)
#             data_summary[time_key] += emissions
#         return render_template('report.html', data_summary=data_summary, total_emissions=total_emissions, country=country_name, timeframe=form.timeframe.data, start_year=form.start_year.data, end_year=form.end_year.data)
#     else:
#             # If there is a validation error, the form will be rendered with error messages
#             for fieldName, errorMessages in form.errors.items():
#                 for err in errorMessages:
#                     flash(f'Error in {fieldName}: {err}', 'danger')
#     return render_template('reportform.html', form=form)


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


@lru_cache(maxsize=1)
def fetch_country_codes():
    url = 'https://restcountries.com/v3.1/all'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return {country['cca3']: country['name']['common'] for country in data}
    return {}

def get_country_name_by_code(country_code):
    countries = fetch_country_codes()
    return countries.get(country_code, "Unknown country code")

def fetch_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        return ET.fromstring(response.content)
    return None

@app.route('/emissions-ranking', methods=['GET', 'POST'])
def country_ranking():
    current_year = datetime.now().year
    form = EmissionRankingForm()
    if request.method == 'GET':
        form.start_year.data = current_year - 1
        form.start_month.data = '1'
        form.end_month.data = '12'
    
    if form.validate_on_submit() or request.method == 'GET':
        start_year = form.start_year.data
        start_month = form.start_month.data
        end_year = start_year  # Assume same year for simplicity
        end_month = form.end_month.data

        base_url = "https://sdmx.oecd.org/public/rest/data"
        dataflow = "OECD.SDD.NAD.SEEA,DSD_AIR_TRANSPORT@DF_AIR_TRANSPORT,1.0"
        time_suffix = ".M......."
        start_period = f"{start_year}-{start_month:02}"
        end_period = f"{end_year}-{end_month:02}"
        url = f"{base_url}/{dataflow}/{time_suffix}?startPeriod={start_period}&endPeriod={end_period}&dimensionAtObservation=AllDimensions"
        print(url)
        root = fetch_data(url)
        country_data = {}
        total_emissions = 0
        if root:
            for obs in root.findall('.//generic:Obs', namespaces={'generic': 'http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic'}):
                country_code = obs.find('.//generic:Value[@id="REF_AREA"]', namespaces={'generic': 'http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic'}).attrib['value']
                emissions = float(obs.find('.//generic:ObsValue', namespaces={'generic': 'http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic'}).attrib['value'])
                country_data[country_code] = country_data.get(country_code, 0) + emissions
                total_emissions += emissions

            emissions_values = list(country_data.values())
            low_threshold, high_threshold = np.percentile(emissions_values, [33, 66])
            
            formatted_countries = []
            for code, emissions in country_data.items():
                percentage = (emissions / total_emissions) * 100
                if emissions < low_threshold:
                    color_class = 'low-emissions'
                elif emissions < high_threshold:
                    color_class = 'medium-emissions'
                else:
                    color_class = 'high-emissions'
                
                formatted_countries.append((get_country_name_by_code(code), format(emissions, ',.2f'), format(percentage, '.2f'), color_class))

            sorted_countries = sorted(formatted_countries, key=lambda item: float(item[1].replace(',', '')), reverse=(form.order.data == 'descending'))

            average_emissions = total_emissions / len(country_data) if country_data else 0

            return render_template('ranking.html', form=form, countries=sorted_countries,
                                   total_emissions=format(total_emissions, ',.2f'), 
                                   average_emissions=format(average_emissions, ',.2f'),
                                   start_year=start_year, start_month=start_month, 
                                   end_year=end_year, end_month=end_month)

    return render_template('ranking.html', form=form)

# @app.route('/emissions-ranking', methods=['GET', 'POST'])
# def country_ranking():
#     current_year = datetime.now().year
#     form = EmissionRankingForm()
#     if request.method == 'GET':
#         form.start_year.data = current_year - 1
#         form.start_month.data = '1'
#         form.end_month.data = '12'
    
#     if form.validate_on_submit() or request.method == 'GET':
#         start_year = form.start_year.data
#         start_month = form.start_month.data
#         end_year = start_year  # Assume same year for simplicity
#         end_month = form.end_month.data
#         order = form.order.data

#         base_url = "https://sdmx.oecd.org/public/rest/data"
#         dataflow = "OECD.SDD.NAD.SEEA,DSD_AIR_TRANSPORT@DF_AIR_TRANSPORT,1.0"
#         time_suffix = ".M......."
#         start_period = f"{start_year}-{start_month:02}"
#         end_period = f"{end_year}-{end_month:02}"
#         url = f"{base_url}/{dataflow}/{time_suffix}?startPeriod={start_period}&endPeriod={end_period}&dimensionAtObservation=AllDimensions"

#         root = fetch_data(url)
#         country_data = {}
#         total_emissions = 0
#         if root:
#             for obs in root.findall('.//generic:Obs', namespaces={'generic': 'http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic'}):
#                 country_code = obs.find('.//generic:Value[@id="REF_AREA"]', namespaces={'generic': 'http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic'}).attrib['value']
#                 emissions = float(obs.find('.//generic:ObsValue', namespaces={'generic': 'http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic'}).attrib['value'])
#                 country_data[country_code] = country_data.get(country_code, 0) + emissions
#                 total_emissions += emissions

#             formatted_countries = [
#                 (get_country_name_by_code(code), format(emissions, ',.2f'), format((emissions / total_emissions) * 100, '.2f'))
#                 for code, emissions in country_data.items()
#             ]

#             sorted_countries = sorted(
#                 formatted_countries,
#                 key=lambda item: float(item[1].replace(',', '')),
#                 reverse=(order == 'descending')
#             )
#         else:
#             sorted_countries = []
#             flash('No data available or an error occurred with the request.', 'warning')

#         average_emissions = total_emissions / len(country_data) if country_data else 0

#         return render_template('ranking.html', form=form, countries=sorted_countries,
#                                total_emissions=format(total_emissions, ',.2f'), 
#                                average_emissions=format(average_emissions, ',.2f'),
#                                start_year=start_year, start_month=start_month, 
#                                end_year=end_year, end_month=end_month)

#     return render_template('ranking.html', form=form)


# @app.route('/emissions-ranking', methods=['GET', 'POST'])
# def country_ranking():
#     current_year = datetime.now().year
#     form = EmissionRankingForm()
#     if request.method == 'GET':
#         form.start_year.data = current_year - 1
#         form.start_month.data = '1'
#         # form.end_year.data = current_year - 1
#         form.end_month.data = '12'
    
#     if form.validate_on_submit() or request.method == 'GET':
#         start_year = form.start_year.data
#         start_month = form.start_month.data
#         # end_year = form.end_year.data
#         end_year = form.start_year.data
#         end_month = form.end_month.data
#         order = form.order.data

#         base_url = "https://sdmx.oecd.org/public/rest/data"
#         dataflow = "OECD.SDD.NAD.SEEA,DSD_AIR_TRANSPORT@DF_AIR_TRANSPORT,1.0"
#         time_suffix = ".M......."
#         start_period = f"{start_year}-{start_month:02}"
#         end_period = f"{end_year}-{end_month:02}"
#         url = f"{base_url}/{dataflow}/{time_suffix}?startPeriod={start_period}&endPeriod={end_period}&dimensionAtObservation=AllDimensions"

#         print(url)
#         root = fetch_data(url)
#         country_data = {}
#         if root:
#             for obs in root.findall('.//generic:Obs', namespaces={'generic': 'http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic'}):
#                 country_code = obs.find('.//generic:Value[@id="REF_AREA"]', namespaces={'generic': 'http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic'}).attrib['value']
#                 emissions = float(obs.find('.//generic:ObsValue', namespaces={'generic': 'http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic'}).attrib['value'])
#                 country_data[country_code] = country_data.get(country_code, 0) + emissions

#             formatted_countries = [
#                 (get_country_name_by_code(code), format(emissions, ',.2f'))
#                 for code, emissions in country_data.items()
#             ]

#             sorted_countries = sorted(
#                 formatted_countries,
#                 key=lambda item: float(item[1].replace(',', '')),
#                 reverse=(order == 'descending')
#             )
#         else:
#             sorted_countries = []
#             flash('No data available or an error occurred with the request.', 'warning')
        
#         return render_template('ranking.html', form=form, countries=sorted_countries, 
#                                start_year=start_year, start_month=start_month, 
#                                end_year=end_year, end_month=end_month)

#     return render_template('ranking.html', form=form)


# @app.route('/emissions-ranking', methods=['GET', 'POST'])
# def country_ranking():
#     current_year = datetime.now().year
#     form = EmissionRankingForm()
#     if request.method == 'GET':
#         form.start_year.data = current_year - 1
#         form.start_month.data = '1'
#         form.end_year.data = current_year - 1
#         form.end_month.data = '12'
    
#     if form.validate_on_submit() or request.method == 'GET':
#         start_year = form.start_year.data
#         start_month = form.start_month.data
#         end_year = form.end_year.data
#         end_month = form.end_month.data
#         order = form.order.data

#         base_url = "https://sdmx.oecd.org/public/rest/data"
#         dataflow = "OECD.SDD.NAD.SEEA,DSD_AIR_TRANSPORT@DF_AIR_TRANSPORT,1.0"
#         time_suffix = ".M......."
#         start_period = f"{start_year}-{start_month:02}"
#         end_period = f"{end_year}-{end_month:02}"
#         url = f"{base_url}/{dataflow}/{time_suffix}?startPeriod={start_period}&endPeriod={end_period}&dimensionAtObservation=AllDimensions"

#         root = fetch_data(url)
#         country_data = {}
#         if root:
#             for obs in root.findall('.//generic:Obs', namespaces={'generic': 'http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic'}):
#                 country_code = obs.find('.//generic:Value[@id="REF_AREA"]', namespaces={'generic': 'http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic'}).attrib['value']
#                 emissions = float(obs.find('.//generic:ObsValue', namespaces={'generic': 'http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic'}).attrib['value'])
#                 country_data[country_code] = country_data.get(country_code, 0) + emissions

#             sorted_countries = sorted(
#                 ((get_country_name_by_code(code), emissions) for code, emissions in country_data.items()),
#                 key=lambda item: item[1],
#                 reverse=(order == 'descending')
#             )
#         else:
#             sorted_countries = []
#             flash('No data available or an error occurred with the request.', 'warning')

#         return render_template('ranking.html', form=form, countries=sorted_countries)

#     return render_template('ranking.html', form=form)



if __name__ == '__main__':
    app.run(debug=True)

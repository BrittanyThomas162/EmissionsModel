from flask_wtf import FlaskForm
from wtforms import StringField, FloatField, SubmitField, SelectField, IntegerField
from wtforms.validators import InputRequired, Length, NumberRange, DataRequired
from utils import fetch_country_codes

class FuelPredictionForm(FlaskForm):
    airline_iata = StringField('Airline IATA Code', validators=[InputRequired(), Length(min=2, max=3)], description="Enter the IATA code of the operator.")
    acft_icao = StringField('Aircraft ICAO Code', validators=[InputRequired(), Length(min=3, max=4)], description="Enter the ICAO code of the aircraft type.")
    acft_class = SelectField('Aircraft Class', choices=[('', 'Choose aircraft class identifier...'),('WB', 'Wide Body'), ('NB', 'Narrow Body'), ('RJ', 'Regional Jet'), ('PJ', 'Private Jet'), ('TP', 'Turbo Propeller'), ('PP', 'Piston Propeller'), ('HE', 'Helicopter'), ('OTHER', 'Other')], validators=[DataRequired()])
    seymour_proxy = StringField('Seymour Proxy Aircraft Code', validators=[InputRequired()], description="Enter the aircraft code for Seymour Surrogate.")
    seats = FloatField('Number of Seats', validators=[InputRequired(), NumberRange(min=1)], description="Enter the number of seats available for the data entry.")
    n_flights = FloatField('Number of Flights', validators=[InputRequired(), NumberRange(min=0)], description="Enter the number of flights of the data entry.")
    iata_departure = StringField('Departure Airport IATA Code', validators=[InputRequired(), Length(min=3, max=3)], description="Enter the IATA code of the origin airport.")
    iata_arrival = StringField('Arrival Airport IATA Code', validators=[InputRequired(), Length(min=3, max=3)], description="Enter the IATA code of the destination airport.")
    distance_km = FloatField('Flight Distance (km)', validators=[InputRequired(), NumberRange(min=1)], description="Enter the flight distance in kilometers.")
    rpk = FloatField('Revenue Passenger Kilometres', validators=[InputRequired(), NumberRange(min=1)], description="Enter the Revenue Passenger Kilometres.")
    fuel_burn_seymour = FloatField('Fuel Burn per Flight (kg)', validators=[InputRequired(), NumberRange(min=0)], description="Enter the fuel burn per flight in kg when seymour proxy available.")
    fuel_burn = FloatField('Total Fuel Burn (kg)', validators=[InputRequired(), NumberRange(min=0)], description="Enter the total fuel burn of the data entry in kg.")
    submit = SubmitField('Predict')


class EmissionForm(FlaskForm):
    country = SelectField('Country', choices=[(code, name) for code, name in fetch_country_codes().items()],
                          validators=[DataRequired()], render_kw={"class": "select2-enable"})
    timeframe = SelectField('Timeframe', choices=[('annual', 'Annual'), ('monthly', 'Monthly'), ('quarterly', 'Quarterly')],
                            validators=[DataRequired()])
    start_year = IntegerField('Start Year', validators=[DataRequired(), NumberRange(min=2013, message="The earliest start year allowed is 2013.")])
    month = SelectField('Month', choices=[(str(i), str(i).zfill(2)) for i in range(1, 13)],
                        validators=[DataRequired()], default=None)
    quarter = SelectField('Quarter', choices=[(str(i), 'Q' + str(i)) for i in range(1, 5)],
                          validators=[DataRequired()], default=None)
    end_year = IntegerField('End Year', validators=[DataRequired()])
    end_month = SelectField('Month', choices=[(str(i), str(i).zfill(2)) for i in range(1, 13)],
                            validators=[DataRequired()], default=None)
    end_quarter = SelectField('Quarter', choices=[(str(i), 'Q' + str(i)) for i in range(1, 5)],
                              validators=[DataRequired()], default=None)
    submit = SubmitField('Generate Report')


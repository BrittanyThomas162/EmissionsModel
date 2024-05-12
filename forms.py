from flask_wtf import FlaskForm
from wtforms import StringField, FloatField, SubmitField, SelectField, IntegerField
from wtforms.validators import InputRequired, Length, NumberRange, DataRequired, ValidationError
from utils import fetch_country_codes

# class FuelPredictionForm(FlaskForm):
#     airline_iata = StringField('Airline IATA Code', validators=[InputRequired(), Length(min=2, max=3)], description="Enter the IATA code of the operator.")
#     acft_icao = StringField('Aircraft ICAO Code', validators=[InputRequired(), Length(min=3, max=4)], description="Enter the ICAO code of the aircraft type.")
#     seats = FloatField('Number of Filled Seats', validators=[InputRequired(), NumberRange(min=1)], description="Enter the number of seats available for the data entry.")
#     n_flights = FloatField('Number of Flights in Entry', validators=[InputRequired(), NumberRange(min=0)], description="Enter the number of flights of the data entry.")
#     iata_departure = StringField('Departure Airport IATA Code', validators=[InputRequired(), Length(min=3, max=3)], description="Enter the IATA code of the origin airport.")
#     iata_arrival = StringField('Arrival Airport IATA Code', validators=[InputRequired(), Length(min=3, max=3)], description="Enter the IATA code of the destination airport.")
#     fuel_burn_seymour = FloatField('Fuel Burn per Flight (kg)', validators=[InputRequired(), NumberRange(min=0)], description="Enter the fuel burn per flight in kg when seymour proxy available.")
#     submit = SubmitField('Predict')

class FuelPredictionForm(FlaskForm):
    airline_iata = StringField('Airline IATA Code', validators=[InputRequired(), Length(min=2, max=3)], description="Enter the IATA code of the operator.")
    acft_icao = StringField('Aircraft ICAO Code', validators=[InputRequired(), Length(min=3, max=4)], description="Enter the ICAO code of the aircraft type.")
    seats = FloatField('Number of Filled Seats', validators=[InputRequired(), NumberRange(min=1)], description="Enter the number of seats available for the data entry.")
    n_flights = FloatField('Number of Flights in Entry', validators=[InputRequired(), NumberRange(min=0)], description="Enter the number of flights of the data entry.")
    # iata_departure = StringField('Departure Airport IATA Code', validators=[InputRequired(), Length(min=3, max=3)], description="Enter the IATA code of the origin airport.")
    # iata_arrival = StringField('Arrival Airport IATA Code', validators=[InputRequired(), Length(min=3, max=3)], description="Enter the IATA code of the destination airport.")
    icao_departure = StringField('Departure Airport ICAO Code', validators=[InputRequired(), Length(min=4, max=4)], description="Enter the ICAO code of the origin airport.")
    icao_arrival = StringField('Arrival Airport ICAO Code', validators=[InputRequired(), Length(min=4, max=4)], description="Enter the ICAO code of the destination airport.")
    fuel_burn_seymour = FloatField('Fuel Burn per Flight (kg)', validators=[InputRequired(), NumberRange(min=0)], description="Enter the fuel burn per flight in kg.")
    submit = SubmitField('Predict')

class EmissionForm(FlaskForm):
    country = SelectField('Country', choices=[(code, name) for code, name in fetch_country_codes().items()],
                          validators=[DataRequired()], render_kw={"class": "select2-enable"})
    timeframe = SelectField('Timeframe', choices=[('annual', 'Annual'), ('monthly', 'Monthly'), ('quarterly', 'Quarterly')],
                            validators=[DataRequired()])
    start_year = IntegerField('Start Year', validators=[DataRequired(), NumberRange(min=2013, message="The earliest start year allowed is for an annual timeframe is 2013.")])
    start_month = SelectField('Start Month', choices=[(str(i), str(i).zfill(2)) for i in range(1, 13)],
                        validators=[DataRequired()], default=None)
    start_quarter = SelectField('Start Quarter', choices=[(str(i), 'Q' + str(i)) for i in range(1, 5)],
                          validators=[DataRequired()], default=None)
    end_year = IntegerField('End Year', validators=[DataRequired()])
    end_month = SelectField('End Month', choices=[(str(i), str(i).zfill(2)) for i in range(1, 13)],
                            validators=[DataRequired()], default=None)
    end_quarter = SelectField('End Quarter', choices=[(str(i), 'Q' + str(i)) for i in range(1, 5)],
                              validators=[DataRequired()], default=None)
    submit = SubmitField('Generate Report')

    def validate_start_year(self, field):
        timeframe = self.timeframe.data
        if timeframe == 'monthly' or timeframe == 'quarterly':
            if field.data < 2019:
                raise ValidationError("For monthly or quarterly timeframes, the earliest start year allowed is 2019.")
       


class EmissionRankingForm(FlaskForm):
    start_year = IntegerField('Year', validators=[NumberRange(min=2019)])
    start_month = SelectField('Start Month', choices=[(str(i), str(i).zfill(2)) for i in range(1, 13)])
    # end_year = IntegerField('End Year', validators=[NumberRange(min=2013)])
    end_month = SelectField('End Month', choices=[(str(i), str(i).zfill(2)) for i in range(1, 13)])
    order = SelectField('Order', choices=[('ascending', 'Ascending'), ('descending', 'Descending')], default='descending')
    submit = SubmitField('Get Ranking')

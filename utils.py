import requests

def fetch_country_codes():
    url = 'https://restcountries.com/v3.1/all'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        countries = {country['cca3']: country['name']['common'] for country in data}
        # Sorting countries by their common name
        sorted_countries = dict(sorted(countries.items(), key=lambda item: item[1]))
        return sorted_countries
    else:
        print("Failed to fetch data")
        return {}

def get_country_name_by_code(country_code):
    # Fetch the dictionary of country codes
    countries = fetch_country_codes()
    # Return the country name matching the country code
    return countries.get(country_code, "Unknown country code")



CREATE TABLE aircraft_data (
    model TEXT,
    icao_code TEXT PRIMARY KEY,
    iata_code TEXT,
    avg_seats NUMERIC,
    avg_sector_km NUMERIC,
    avg_fuel_burn_kg_km NUMERIC,
    avg_fuel_per_seat_l_100km NUMERIC
);

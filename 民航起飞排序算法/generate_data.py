# ----------------------
# generate_data.py
# ----------------------
import numpy as np
import random
import pandas as pd


def generate_flight_data(n_flights=100, time_window=180):
    np.random.seed(42)
    intervals = np.random.exponential(scale=2.0, size=n_flights)
    planned_times = np.cumsum(intervals)
    planned_times = np.mod(planned_times, time_window)

    flights = []
    for i in range(n_flights):
        flight_id = f"CA{i // 1000:02d}{i % 1000:04d}"
        delay = max(0, int(np.random.normal(15, 8)))
        actual_arrival = planned_times[i] + delay

        aircraft_type = random.choice(['A320', 'B737', 'A350'])
        if aircraft_type == 'A320':
            fuel_coeff = 1.0
            vortex_interval = 2
        elif aircraft_type == 'B737':
            fuel_coeff = 1.2
            vortex_interval = 3
        else:
            fuel_coeff = 0.8
            vortex_interval = 4

        flight_type = random.choice(['normal', 'VIP', 'emergency'])
        delay_weight = 1.0 if flight_type == 'normal' else 2.0 if flight_type == 'VIP' else 3.0

        flights.append({
            'id': flight_id,
            'planned': int(planned_times[i]),
            'actual': int(actual_arrival),
            'fuel': fuel_coeff,
            'vortex': vortex_interval,
            'weight': delay_weight,
            'aircraft_type': aircraft_type,
            'flight_type': flight_type
        })
    return flights


def save_to_csv(flights, filename="flight_data.csv"):
    df = pd.DataFrame(flights)
    df.to_csv(filename, index=False)


if __name__ == "__main__":
    flights = generate_flight_data(n_flights=150)
    save_to_csv(flights, "flight_data.csv")
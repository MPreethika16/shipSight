"""Configuration constants, corridor definitions, and schema definitions for ShipSight."""

RANDOM_SEED = 42
NUM_SHIPMENTS = 250
TRAIN_RATIO = 0.80
DELAY_RISK_THRESHOLD_HOURS = 6.0

# 5 Major Intermodal Shipping Corridors
CORRIDORS = {
    "CNSHA-USLAX": {
        "origin": "Shanghai (CNSHA)",
        "destination_port": "Port of Los Angeles (USLAX)",
        "inland_hub": "Ontario Logistics Center, CA",
        "distance_nm": 5700.0,
        "nominal_speed_kts": 19.5,
        "sched_berth_hours": 16.0,
        "sched_terminal_hours": 36.0,
        "sched_customs_hours": 18.0,
        "sched_drayage_hours": 12.0,
    },
    "SGSIN-NLRTM": {
        "origin": "Singapore (SGSIN)",
        "destination_port": "Port of Rotterdam (NLRTM)",
        "inland_hub": "Duisburg Intermodal Hub, Germany",
        "distance_nm": 8300.0,
        "nominal_speed_kts": 20.0,
        "sched_berth_hours": 18.0,
        "sched_terminal_hours": 40.0,
        "sched_customs_hours": 20.0,
        "sched_drayage_hours": 18.0,
    },
    "BEANR-USNYC": {
        "origin": "Antwerp (BEANR)",
        "destination_port": "Port of New York & New Jersey (USNYC)",
        "inland_hub": "Bethlehem Logistics Center, PA",
        "distance_nm": 3450.0,
        "nominal_speed_kts": 19.0,
        "sched_berth_hours": 14.0,
        "sched_terminal_hours": 32.0,
        "sched_customs_hours": 16.0,
        "sched_drayage_hours": 10.0,
    },
    "KRPUS-JPYOK": {
        "origin": "Busan (KRPUS)",
        "destination_port": "Yokohama (JPYOK)",
        "inland_hub": "Saitama Distribution Hub, Japan",
        "distance_nm": 620.0,
        "nominal_speed_kts": 16.0,
        "sched_berth_hours": 12.0,
        "sched_terminal_hours": 24.0,
        "sched_customs_hours": 12.0,
        "sched_drayage_hours": 8.0,
    },
    "INNSA-GBFXT": {
        "origin": "Nhava Sheva (INNSA)",
        "destination_port": "Port of Felixstowe (GBFXT)",
        "inland_hub": "Northampton Logistics Park, UK",
        "distance_nm": 6400.0,
        "nominal_speed_kts": 19.0,
        "sched_berth_hours": 16.0,
        "sched_terminal_hours": 36.0,
        "sched_customs_hours": 18.0,
        "sched_drayage_hours": 14.0,
    },
}

CARRIERS = ["MSK", "CMA", "COS", "ONE"]
CARGO_TYPES = ["General", "Reefer", "Hazmat"]

# Whitelisted features strictly available at prediction cutoff t_pred = actual_berth_time
FEATURE_WHITELIST = [
    # Static Corridor & Shipment Attributes
    "route_id",
    "carrier_code",
    "cargo_type",
    "container_teu",
    "transit_distance_nm",
    # Planned Downstream Allowances (Scheduled buffers)
    "scheduled_terminal_hours",
    "scheduled_customs_hours",
    "scheduled_drayage_hours",
    # Upstream Realized Metrics & Manifest Data Known at t_pred
    "weather_severity_index",
    "carried_delay_at_anchorage",
    "port_congestion_index",
    "carried_delay_at_berth",
    "customs_inspection_flag",
]

# Blacklist of prohibited downstream features and ground-truth targets
PROHIBITED_LEAKAGE_COLUMNS = [
    "actual_terminal_hours",
    "actual_terminal_end",
    "actual_customs_hours",
    "actual_customs_end",
    "actual_drayage_hours",
    "actual_final_delivery_time",
    "remaining_duration_hours",
    "final_delay_hours",
    "delay_risk_binary",
    "shipment_id",
]

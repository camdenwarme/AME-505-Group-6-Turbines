"""Train MAPPO on the six-turbine OFF/FLORIS layout."""

from train_mappo import main


# Positions in rotor diameters, in the same order as the six-turbine OFF case.
SIX_TURBINE_LAYOUT = (
    (0.0, -2.5), (5.0, -2.5), (10.0, -2.5),
    (0.0, 2.5), (5.0, 2.5), (10.0, 2.5),
)


if __name__ == "__main__":
    main(default_layout=SIX_TURBINE_LAYOUT)

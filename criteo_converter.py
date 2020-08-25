import csv


# pass two: write new file
with open("day_0") as fcsv:
    for day in range(24):
        with open('input/day_'+str(day), 'w') as f:
            for i in range(5000000):
                f.write(fcsv.readline())


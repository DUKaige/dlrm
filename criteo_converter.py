import csv


# pass two: write new file
with open("/Users/liukaige/Downloads/day_0") as fcsv:
    for day in range(24):
        with open('input/day_'+str(day), 'w') as f:
            for i in range(50000):
                f.write(fcsv.readline())


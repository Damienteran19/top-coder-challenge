import sys

if (len(sys.argv) == 4):
    total_days = float(sys.argv[1])
    total_miles = float(sys.argv[2])
    total_reciept = float(sys.argv[3])

    reimburse = (100 * total_days) + (0.58 * total_miles) + (1.00 * total_reciept)
    sys.stdout.write(str(reimburse))
else:
    sys.stderr.write("Incorrct argument count\nUsage: python predict.y [days] [miles] [receipt]")

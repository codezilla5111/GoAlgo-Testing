### TO BE RUN ONLY ONCE! ###
from main import db,Example
import csv

db.create_all()

with open("data.csv") as f:
    reader = csv.reader(f)
    header = next(reader)
    for i in reader:
        new_entry = Example(
        title = i[1],
        keywords = i[2],
        url = i[3],
        html=i[4])
        db.session.add(new_entry)
        db.session.commit()

import matplotlib.pyplot as plt

labels = 'Medical Waste', 'E-Waste', 'Glass', 'Plastic', 'Cardboard', 'Paper', 'Metal'
sizes = [11.68, 14.28, 14.69, 14.57, 14.09, 15.66, 15.03]

fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.2f%%')
fig.tight_layout()
fig.savefig('data_dist.png')
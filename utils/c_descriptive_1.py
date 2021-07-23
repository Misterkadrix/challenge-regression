from utils.b_preprocessing import preprocess
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-muted')

ds = preprocess()

plt.boxplot(ds.room_number)
plt.title('Room number')
plt.show()
plt.boxplot(ds.price)
plt.title('Price in €')
plt.show()
plt.boxplot(ds.area)
plt.title('Area in m²')
plt.show()
plt.boxplot(ds.land_surface)
plt.title('Land surface in m²')
plt.show()

'''
plt.title('# Properties in dataset, by Regions')
plt.pie(ds.Region.value_counts(normalize=True).values,
        labels=ds.Region.value_counts(normalize=True).index.values, startangle=90, autopct='%1.1f%%')
plt.savefig("assets/" + datetime.now().strftime("%Y%m%d_%I%M%S%p") + "_" + "properties_region.png", transparent=True)
plt.show()

plt.title('Type of properties')
plt.pie(ds.type.value_counts(normalize=True).values,
        labels=ds.type.value_counts(normalize=True).index.values, startangle=90, autopct='%1.1f%%')
plt.savefig("assets/" + datetime.now().strftime("%Y%m%d_%I%M%S%p") + "_" + "type_properties.png", transparent=True)
plt.show()

plt.title('Subtype of properties')
plt.pie(ds.subtype.value_counts(normalize=True).values,
        labels=ds.subtype.value_counts(normalize=True).index.values, startangle=90, autopct='%1.1f%%')
plt.savefig("assets/" + datetime.now().strftime("%Y%m%d_%I%M%S%p") + "_" + "subtype_properties.png", transparent=True)
plt.show()

''
plt.title('Equipped kitchen or not')
plt.pie(ds.FullyEquippedKitchen.value_counts(normalize=True).values,
        labels=ds.FullyEquippedKitchen.value_counts(normalize=True).index.values, startangle=90, autopct='%1.1f%%')
#plt.legend(loc='lower right')
plt.savefig("assets/" + datetime.now().strftime("%Y%m%d_%I%M%S%p") + "_" + "kitchen_install.png", transparent=True)
plt.show()
''

fig, ax = plt.subplots()
ax.xaxis.set_major_formatter('€{x:1.1f}M')

sns.histplot(ds.price*1e-6, kde=True)#, bins=100)
#plt.axis([0,600,0,400])
plt.xlabel('Price in M€')
plt.title('Distribution of the price of properties in dataset')
plt.grid(True)
plt.savefig("assets/" + datetime.now().strftime("%Y%m%d_%I%M%S%p") + "_" + "price_distribution.png", transparent=True)
plt.show()

sns.histplot(ds.price, kde=True, log_scale=True)#, bins=100)
#plt.axis([0,600,0,400])
plt.xlabel('Price in € (Log scale)')
plt.title('Distribution of the price of properties in dataset - Log scale')
plt.grid(True)
plt.savefig("assets/" + datetime.now().strftime("%Y%m%d_%I%M%S%p") + "_" + "price_distribution_log.png", transparent=True)
plt.show()

sns.histplot(ds.priceSqMeter, kde=True)#, bins=100)
#plt.axis([0,600,0,400])
plt.xlabel('Price/m² in €')
plt.title('Distribution of the price/m²')
plt.grid(True)
plt.savefig("assets/" + datetime.now().strftime("%Y%m%d_%I%M%S%p") + "_" + "price_sqmeter_distribution.png", transparent=True)
plt.show()
'''

from utils.b_preprocessing import preprocess
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-muted')

ds = preprocess()
#ds_corr = ds.iloc[:, [3, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18, 19, 25, 26, 27]]  # quickly selecting numerical variables

#sns.heatmap(ds_corr.corr(), annot=True, fmt='.1g', vmin=0, vmax=1, center=0.3, cmap= 'coolwarm')
sns.heatmap(ds.corr(), annot=True, fmt='.1g', vmin=0, vmax=1, center=0.3, cmap= 'coolwarm')
plt.savefig("assets/" + datetime.now().strftime("%Y%m%d_%I%M%S%p") + "_" + "correlations.png", transparent=True)
plt.title('Correlations after dropping duplicates zip/price/area/bedrooms and missing price or area')
plt.show()

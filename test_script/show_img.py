import matplotlib.pyplot as plt
from PIL import Image
img = Image.open('/media/arnold/新加卷/Data/trackdata/B-folder/outputs/B_mot2_10_0.3_nms_ZX/Track2/00010.jpg')
fig = plt.figure()
ax1= fig.add_subplot(111)
ax1.imshow(img)
plt.show()

# load and evaluate a saved model
from matplotlib import pyplot as plt
from keras.models import load_model
import numpy as np

# load model
model = load_model('HARModel1D_batchNorm.h5')

# summarize model.
model.summary()

# View output dimensions
# for l in model.layers:
#     plt.imshow(l.output[0])
#     plt.show()
#     break

# View weights
j = 0
for w in model.get_weights():
    print(np.shape(w))
    if len(np.shape(w)) >= 3:
         plt.figure(figsize=(100, 40))
         for i in range(np.shape(w)[2]):
             plt.subplot(8, int(np.shape(w)[2]) / 8, i + 1)
             plt.imshow(w[:, :, i])
             plt.colorbar()
             f = "(%d, %d, %d)_%d.jpg" % (np.shape(w)[0], np.shape(w)[1], np.shape(w)[2], j)
         #plt.savefig(f)
         plt.close()
         j += 1

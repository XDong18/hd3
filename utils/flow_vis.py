import numpy as np
import matplotlib.pyplot as plt
import cv2


def main():
flow_fn = '../predictions/semi_lr_0.001_gap_1_fix/vec/b1c81faa-3df17267/b1c81faa-3df17267-0000002.png'
img_1_fn = '/shared/xudongliu/code/semi-flow/hd3/semi_color_mask_lr_0.001/frame_1/b1c81faa-3df17267/b1c81faa-3df17267-0000003.jpg'
img_0_fn = '/shared/xudongliu/code/semi-flow/hd3/semi_color_mask_lr_0.001/frame_1/b1c81faa-3df17267/b1c81faa-3df17267-0000002.jpg'

flow = cv2.imread(flow_fn, -1)[:, :, ::-1].astype(np.float)
flow = (flow[:, :, :2] - 2.0 ** 15) / 64.0
x_flow = - flow[:,:,0][::-1, :]
y_flow = flow[:,:,1][::-1, :]

img_1 = cv2.imread(img_1_fn)
img_0 = cv2.imread(img_0_fn)
H, W, _ = img_1.shape

x_density = 50
x, y = np.meshgrid(np.linspace(0, W-1, x_density),np.linspace(0, H-1, x_density * (H // W)))
u = cv2.resize(x_flow, (x_density * (H // W), x_density), interpolation=cv2.INTER_NEAREST)
v = cv2.resize(y_flow, (x_density * (H // W), x_density), interpolation=cv2.INTER_NEAREST)
plt.imshow(img_1)
plt.quiver(x, y, u, v, color=(255, 255, 255))
plt.show()

if __name__=='__main__':
    main()
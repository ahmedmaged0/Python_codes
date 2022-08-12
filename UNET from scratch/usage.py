# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 15:40:20 2022

@author: ahmed
"""
#IMAGE[1] 

valid_data1 = UNETDataset(IMAGE, MSK, validation_aug)


test = A.Resize(HEIGHT, WIDTH)
test
dir_test = 'C:/Users/ahmed/Desktop/Water Marked/Island-Photo_Watermark.jpg'
dir_test = 'C:/Users/ahmed/Desktop/Water Marked/Screenshot.jpg'
image = cv2.imread(dir_test)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = np.array(image)
plt.imshow(image)



transformed = test(image=image)
transformed_image = transformed['image']
plt.imshow(transformed_image)

UNETmodel.eval()

transformed_image2 = torch.from_numpy(transformed_image).permute(2,0,1)/255
with torch.no_grad():
    test_msk = UNETmodel(transformed_image2.unsqueeze(0))

test_msk = test_msk.squeeze(0).detach().numpy()
test_msk = test_msk.squeeze(0)
plt.imshow(test_msk,cmap='gray')






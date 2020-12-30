from tensorflow.keras.preprocessing.image import load_img

for img_name in range(1, 17):
    img_name = 'image' + str(img_name)
    print(img_name)

    img = 'images/' + img_name + '_original.jpg'

    img = load_img(img)

    size = img.size

    x = int(size[0] / 2)
    y = int(size[1] / 2)

    img = img.resize((x, y))

    img.save('images/' + img_name + '_lowres.png')

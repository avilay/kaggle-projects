## Feature Engineering
I have downsized the photos from their original dimensions of 1365x2048 to 250x250. If I don't get good results with this I can increase the size.

Another idea for pre-processing is to that the leaf is usually in focus of the camera, but there is a lot of other foliage that is slightly out of focus. I can figure out a way to crop the photo around the leaf. That might increase the performance. The argument against this is that the model will learn to ignore these areas. This argument is only true if there is sufficient amount of data, which I don't have. So resorting to such manual feature engineering might be the right thing to do.


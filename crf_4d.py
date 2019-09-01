def refine_softmax_crf(smArray, imageArray, use_2d, n_iter, sdims_in, schan_in, compat_in):

    import numpy as np
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian

    imageArray = np.float32(imageArray)
    n_labels = smArray.shape[0]
    labelsArray2 = np.zeros_like(imageArray)

    if use_2d:
	print("Perform fully connected CRF (softmax) - 2D mode slice-by-slice")
        for slice in range(0, imageArray.shape[2]):
            image = imageArray[:,:,slice] * 255
            image = np.tile(image,(3,1,1))
            image = np.moveaxis(image, 0, 2).astype('uint8')

            sm = smArray[:,:,slice,:]

            # Example using the DenseCRF class and the util functions
            d = dcrf.DenseCRF(image.shape[0] * image.shape[1], n_labels) #?

            # get unary potentials (neg log probability)
            U = unary_from_softmax(np.reshape(sm.transpose(2,0,1), (n_labels,-1))) #?
            d.setUnaryEnergy(U)

            # This creates the color-independent features and then add them to the CRF
            feats = create_pairwise_gaussian(sdims=(3, 3), shape=image.shape[:2])
            d.addPairwiseEnergy(feats, compat=3,
                                kernel=dcrf.DIAG_KERNEL,
                                normalization=dcrf.NORMALIZE_SYMMETRIC)

            # This creates the color-dependent features and then add them to the CRF
            feats = create_pairwise_bilateral(sdims=(5, 5), schan=(5, 5, 5),
                                              img=image, chdim=2)
            d.addPairwiseEnergy(feats, compat=10,
                                kernel=dcrf.DIAG_KERNEL,
                                normalization=dcrf.NORMALIZE_SYMMETRIC)

            # Run inference steps.
            Q = d.inference(n_iter)
            labels2 = np.argmax(Q, axis=0).reshape(image.shape[0], image.shape[1])
            labelsArray2[:,:,slice] = labels2
    else:
	print("Perform fully connected CRF (softmax) - 3D mode")

        # image = imageArray * 255.0 / np.max(imageArray)
        # image = np.tile(image, (3, 1, 1, 1))
        # image = np.moveaxis(image, 0, 3).astype('uint8')

        image = imageArray[:, :, :, np.newaxis]

        # Example using the DenseCRF class and the util functions
        d = dcrf.DenseCRF(image.shape[0] * image.shape[1] * image.shape[2], n_labels)

        smArray_reshape = np.reshape(smArray, (n_labels,-1))
        print('smArray_reshape.shape: ',smArray_reshape.shape)
        # get unary potentials (neg log probability)
        U = unary_from_softmax(smArray_reshape) #---------------------
        U = U.copy(order='C') # prevent C-contiguous error #?
        print('U_shape: ', np.array(U).shape)
        d.setUnaryEnergy(U)

        # This creates the color-independent features and then add them to the CRF
        # feats = create_pairwise_gaussian(sdims=(3, 3, 3), shape=image.shape[:3])
        # d.addPairwiseEnergy(feats, compat=3,
        #                     kernel=dcrf.DIAG_KERNEL,
        #                     normalization=dcrf.NORMALIZE_SYMMETRIC)
        # Run inference steps.
        Q = d.inference(n_iter)
        labelsArray2 = np.argmax(Q, axis=0).reshape(image.shape[0], image.shape[1], image.shape[2])

    #    if labelArray.shape != labelsArray2.shape:
     #       print "Shape mismatch 2"
      #  else:
       #     intersection = np.logical_and(labelArray, labelsArray2)
        #    dice = 2.0 * intersection.sum() / (labelArray.sum() + labelsArray2.sum())
         #   print "5iterations: compat_in:", compat_in, " Dice:", dice


        # Q, tmp1, tmp2 = d.startInference()
        # for _ in range(5):
        #     d.stepInference(Q, tmp1, tmp2)
        # kl1 = d.klDivergence(Q) / (image.shape[0] * image.shape[1] * image.shape[2])
        # map_soln1 = np.argmax(Q, axis=0).reshape((image.shape[0], image.shape[1], image.shape[2]))
        #
	# intersection = np.logical_and(labelArray, map_soln1)
        # dice = 2.0 * intersection.sum() / (labelArray.sum() + map_soln1.sum())
        # print "5iterations: compat_in:", compat_in, " Dice:", dice, " KL:", kl1
        #
	# for _ in range(20):
        #     d.stepInference(Q, tmp1, tmp2)
        # kl2 = d.klDivergence(Q) / (image.shape[0] * image.shape[1] * image.shape[2])
        # map_soln2 = np.argmax(Q, axis=0).reshape((image.shape[0], image.shape[1], image.shape[2]))
        #
	# intersection = np.logical_and(labelArray, map_soln2)
        # dice = 2.0 * intersection.sum() / (labelArray.sum() + map_soln2.sum())
        # print "25iterations: Dice: ", dice, "KL:", kl2
        #
	# for _ in range(50):
        #     d.stepInference(Q, tmp1, tmp2)
        # kl3 = d.klDivergence(Q) / (image.shape[0] * image.shape[1] * image.shape[2])
        # map_soln3 = np.argmax(Q, axis=0).reshape((image.shape[0], image.shape[1], image.shape[2]))
        #
	# intersection = np.logical_and(labelArray, map_soln3)
        # dice = 2.0 * intersection.sum() / (labelArray.sum() + map_soln3.sum())
        # print "75iterations: Dice: ", dice, "KL:", kl3

    return labelsArray2

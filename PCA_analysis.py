import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from ImageDataIO import ImageDataIO
from GAN_eval_metrics import ConvNetFeatureSaver_Keras

if __name__ == "__main__":

	save_fig_dir = ""
	
    ind_to_observe = [3,9,15,21,27,33]
    #for ind in range(36):
    for ind in ind_to_observe:
        idio = ImageDataIO(extention="png", is2D=True)

        train_bapl1 = "..\\training\\bapl1\\" + str(ind)
        train_bapl3 = "..\\training\\bapl3\\" + str(ind)

        origin_bapl1 = "..\\original\\bapl1\\"+str(ind)
        origin_bapl3= "..\\original\\bapl3\\"+str(ind)

        gen_bapl1 = "..\\generated\\bapl1\\"+str(ind)
        gen_bapl3 = "..\\generated\\bapl3\\"+str(ind)


        # read training_bapl1
        train_bapl1, train_bapl1_filename = idio.read_files_from_dir(train_bapl1)
        print(len(train_bapl1))
        train_bapl1 = idio._resizing_channel(train_bapl1, resize_size=None, channel_size=3)
        train_bapl1 = idio.convert_PIL_to_numpy(train_bapl1)
        print("train_bapl1.shape",train_bapl1.shape)


        # read training_bapl3
        train_bapl3, train_bapl3_filename = idio.read_files_from_dir(train_bapl3)
        print(len(train_bapl3))
        train_bapl3 = idio._resizing_channel(train_bapl3, resize_size=None, channel_size=3)
        train_bapl3 = idio.convert_PIL_to_numpy(train_bapl3)
        print("train_bapl3.shape", train_bapl3.shape)


        # read original_bapl1
        origin_bapl1, origin_bapl1_filename = idio.read_files_from_dir(origin_bapl1)
        print(len(origin_bapl1))
        origin_bapl1 = idio._resizing_channel(origin_bapl1, resize_size=None, channel_size=3)
        origin_bapl1 = idio.convert_PIL_to_numpy(origin_bapl1)
        print(origin_bapl1.shape)

        # read original_bapl3
        origin_bapl3, origin_bapl3_filename = idio.read_files_from_dir(origin_bapl3)
        print(len(origin_bapl3))
        origin_bapl3 = idio._resizing_channel(origin_bapl3, resize_size=None, channel_size=3)
        origin_bapl3 = idio.convert_PIL_to_numpy(origin_bapl3)
        print(origin_bapl3.shape)

        # read generated_bapl1
        gen_bapl1, gen_bapl1_filename = idio.read_files_from_dir(gen_bapl1)
        print(len(gen_bapl1))
        gen_bapl1 = idio._resizing_channel(gen_bapl1, resize_size=None, channel_size=3)
        gen_bapl1 = idio.convert_PIL_to_numpy(gen_bapl1)
        print(gen_bapl1.shape)

        # read generated_bapl3
        gen_bapl3, gen_bapl3_filename = idio.read_files_from_dir(gen_bapl3)
        print(len(gen_bapl3))
        gen_bapl3 = idio._resizing_channel(gen_bapl3, resize_size=None, channel_size=3)
        gen_bapl3 = idio.convert_PIL_to_numpy(gen_bapl3)
        print(gen_bapl3.shape)

        conv_f_saver = ConvNetFeatureSaver_Keras(model='densenet121', batchSize=64, input_shape=(95, 79, 3))
        train_bapl1_feature = conv_f_saver.feature_extractor_from_npMat(train_bapl1, func=None)
        print("feature space, train_bapl1_feature", train_bapl1_feature.shape)

        train_bapl3_feature = conv_f_saver.feature_extractor_from_npMat(train_bapl3, func=None)
        print("feature space, train_bapl3_feature", train_bapl3_feature.shape)

        origin_bapl1_feature = conv_f_saver.feature_extractor_from_npMat(origin_bapl1, func=None)
        print("feature space", origin_bapl1_feature.shape)

        origin_bapl3_feature = conv_f_saver.feature_extractor_from_npMat(origin_bapl3, func=None)
        print("feature space", origin_bapl3_feature.shape)

        gen_bapl1_feature = conv_f_saver.feature_extractor_from_npMat(gen_bapl1, func=None)
        print("feature space", gen_bapl1_feature.shape)

        gen_bapl3_feature = conv_f_saver.feature_extractor_from_npMat(gen_bapl3, func=None)
        print("feature space", gen_bapl3_feature.shape)


        images_feature = np.concatenate([train_bapl1_feature, train_bapl3_feature, origin_bapl1_feature, origin_bapl3_feature, gen_bapl1_feature, gen_bapl3_feature])
        labels = ["train_bapl1"]*len(train_bapl1_feature) + ["train_bapl3"]*len(train_bapl3_feature)\
            +["original_bapl1"]*len(origin_bapl1_feature) + ["original_bapl3"]*len(origin_bapl3_feature)\
                 +["generated_bapl1"]*len(gen_bapl1_feature) + ["generated_bapl3"]*len(gen_bapl3_feature)

        filenames = np.concatenate([train_bapl1_filename, train_bapl3_filename,origin_bapl1_filename, origin_bapl3_filename, gen_bapl1_filename, gen_bapl3_filename])
        print("images_feature shape", images_feature.shape) 
        #print("images_feature shape", images_feature.shape)  
        print("filenames shape", filenames.shape) # (298,)
        

        labels_sequence = ["train_bapl1","train_bapl3","original_bapl1","original_bapl3","generated_bapl1","generated_bapl3"]
        #label_to_id_dict = {v: i for i, v in enumerate(np.unique(labels))}
        label_to_id_dict = {v: i for i, v in enumerate(labels_sequence)}
        id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}

        label_ids = np.array([label_to_id_dict[x] for x in labels])
        print("label_id shape", label_ids.shape)  # 603

        scaler = StandardScaler()
        images_scaled = scaler.fit_transform(images_feature)
        print("images_scaled", images_scaled.shape)  # (603, 12288)
        print("images_scaled min, max", images_scaled.min(), images_scaled.max())  # (603, 12288)

		# 1. PCA
        # pca = PCA(n_components=2)
        # pca_result = pca.fit_transform(images_scaled)
        # print("pca_result", pca_result.shape)
        # print("sum", sum(pca.explained_variance_ratio_))
        # print("explained_variance_ratio", pca.explained_variance_ratio_ * 100)
        # tsne_result = pca_result

		# 2. t-SNE
        tsne = TSNE(n_components=2, perplexity=40.0)
        tsne_result = tsne.fit_transform(images_scaled)

        tsne_result_scaled = StandardScaler().fit_transform(tsne_result)  # apply StandardScaler again
        print("tsne result scaled", tsne_result_scaled.shape)


        def visualize_scatter(data_2d, label_ids, id_to_label_dict=None, figsize=(20, 20),
                              marker_list=None, filename_list=None, img_save_dir=None, marker_size=None):
            if not id_to_label_dict:
                id_to_label_dict = {v: i for i, v in enumerate(np.unique(label_ids))}

            plt.figure(figsize=figsize)
            plt.grid()

            nb_classes = len(np.unique(label_ids))

            cmap = plt.cm.get_cmap("jet", nb_classes)

            for i, label_id in enumerate(np.unique(label_ids)):
                # if "gen" in label_id:
                #print(label_id) # 0 or 1 or 2 or 3
                if marker_list is not None:
                    ax = plt.scatter(data_2d[np.where(label_ids == label_id), 0],
                                data_2d[np.where(label_ids == label_id), 1],
                                marker=marker_list[i][0],
                                c=marker_list[i][1],
                                linewidth=2.0,
                                alpha=0.8,
                                label=id_to_label_dict[label_id],
                                s=marker_size)

                else :
                    ax = plt.scatter(data_2d[np.where(label_ids == label_id), 0],
                            data_2d[np.where(label_ids == label_id), 1],
                            marker='o',
                            c=cmap(i),
                            linewidth='5',
                            alpha=0.8,
                            label=id_to_label_dict[label_id],
                                s=marker_size)
                if filename_list is not None :
                    x = data_2d[np.where(label_ids == label_id), 0]+0.02
                    y = data_2d[np.where(label_ids == label_id), 1]
                    n = filename_list[np.where(label_ids == label_id)]
                    print("x", x.shape)
                    print("y", y.shape)
                    print("n", n.shape)
                    # for ind in range(len(x)):
                    #     print(ind, x, y, n)
                    for ind in range(len(n)):
                        ax.axes.text(x[0][ind], y[0][ind], n[ind])
                    # ax.axes.text(data_2d[np.where(label_ids == label_id), 0]+0.02, data_2d[np.where(label_ids == label_id), 1],
                    #               filenames[label_ids == label_id])
            # plt.legend(loc='best')
            plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5),
                       fancybox=True, shadow=True, ncol=1, fontsize=figsize[0])
            #plt.axis('off')
            plt.grid(b=None)
            if img_save_dir is not None:
                plt.savefig(img_save_dir)
            else :
                plt.show()



        def visualize_scatter_with_images(data_2d, images, figsize=(45, 45), image_zoom=1):
            fig, ax = plt.subplots(figsize=figsize)
            plt.grid()
            artists = []
            for xy, i in zip(data_2d, images):
                x0, y0 = xy
                img = OffsetImage(i, zoom=image_zoom)
                ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
                artists.append(ax.add_artist(ab))
            ax.update_datalim(data_2d)
            ax.autoscale()
            plt.show()

        marker_list = [('$T$', "black"), ('$T$', "red"),('o', "black"), ('o', "red"), ('+', "black"), ('+', "red")]

        fig_filename = "t_SNE_"
        
		img_save_dir = save_fig_dir+"\\"+fig_filename+str(ind)+".png"
        #img_save_dir = None
        filenames = None
        marker_size=20*2**3
        visualize_scatter(tsne_result_scaled, label_ids, id_to_label_dict, figsize=(10, 10),
                          marker_list=marker_list, filename_list=filenames, img_save_dir=img_save_dir, marker_size=marker_size)

# A set of functions for organoid detection and quantification
# 1- detect_organoids
# Timothy Kassis, last updated 01/22/18

def get_metrics(boxes):
    import numpy as np

    # grab the coordinates of the bounding boxes
    x1 = boxes[:,1]
    y1 = boxes[:,0]
    x2 = boxes[:,3]
    y2 = boxes[:,2]

    box_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    d1 = x2 - x1 + 1
    d2 = y2 - y1 + 1
    org_area = 3.1415 * d1/2 * d2/2

    d_max = np.maximum(d1,d2)
    d_min = np.minimum(d1,d2)
    return(x1, y1, x2, y2, org_area, d_max, d_min)

def nms(boxes, scores, classes): # Adapted from Adrian Rosebrock
    import numpy as np
    overlapThresh = 0.90
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[0][:,1]
    y1 = boxes[0][:,0]
    x2 = boxes[0][:,3]
    y2 = boxes[0][:,2]

    # area of the bounding boxes and sort the bounding boxes by the bottom-right y-coordinate
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        #idxs = np.delete(idxs, np.concatenate(([last],np.where((overlap > overlapThresh) and ('replace with area condition'))[0])))
        idxs = np.delete(idxs, np.concatenate(([last],np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the integer data type
    nms_boxes = boxes[0][pick]
    nms_scores = scores[0][pick]
    nms_classes = classes[0][pick]

    return nms_boxes, nms_scores, nms_classes

def detect_organoids(image_np,
                    PATH_TO_CKPT,
                    PATH_TO_LABELS,
                    NUM_CLASSES = 1,
                    thresh = 0.95):
    # Import required files
    import sys
    sys.path.append("...")
    import numpy as np
    import tensorflow as tf
    from utils import label_map_util
    from core import box_list_ops

    # Load frozen Tensorflow model into memory and label map
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Perform detection
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

    # Filter boxes according to threshold and also remove edge boxes
    [im_height, im_width, depth] = np.shape(image_np)
    # Iterate through boxes and remove ones at edge within 5 pixels
    for i, box in enumerate(boxes[0]):
        # Unnormalize box coordinates
        boxes[0][i] = [box[0]*im_height, box[1]*im_width, box[2]*im_height, box[3]*im_width]
        if (box[0]<5) or (box[1] <5) or (box[2]>im_height-5) or (box[3]>im_width-5):
            scores[0][i] = 0 # Set score of edge boxes to 0

    # Filter low scores below threshold
    high_score_indices = np.reshape(np.where(np.greater(scores, thresh))[1],[-1]).astype(np.int32)
    boxes_filtered = np.take(boxes,high_score_indices,axis=1)
    scores_filtered = np.take(scores,high_score_indices,axis=1)
    classes_filtered = np.take(classes,high_score_indices,axis=1)

    return boxes_filtered, scores_filtered, classes_filtered, num, category_index

def detect_all_organoids(IMAGE_PATH,
                        PATH_TO_CKPT,
                        PATH_TO_LABELS,
                        NUM_CLASSES = 1,
                        look_pix=450,
                        slide_pix=200,
                        thresh = 0.95):
    # Import required files
    import sys
    sys.path.append("..")
    import numpy as np
    from PIL import Image
    from organoid_utils import detect_organoids

    # Open image and load into numpy array
    full_image = Image.open(IMAGE_PATH)
    (full_im_width, full_im_height) = full_image.size
    full_image_np = np.array(full_image.getdata()).reshape((full_im_height, full_im_width, 3)).astype(np.uint8)

    # Pad image
    width_padding = (round((full_im_width+look_pix/2)/look_pix)*look_pix) - full_im_width
    height_padding = (round((full_im_height+look_pix/2)/look_pix)*look_pix) - full_im_height
    padded_im_width = full_im_width + width_padding
    padded_im_height = full_im_height + height_padding
    padded_image = full_image.crop((0, 0, padded_im_width, padded_im_height))
    padded_image_np = np.array(padded_image.getdata()).reshape((padded_im_height, padded_im_width,3)).astype(np.uint8)

    # Perform detection on crops of padded image
    all_boxes = np.empty((0,4)).astype('float32')
    all_scores= np.empty((1,0)).astype('float32')
    all_classes= np.empty((1,0)).astype('float32')

    y_range = range(0,(padded_im_height-look_pix)+1,slide_pix)
    x_range = range(0,(padded_im_width-look_pix)+1,slide_pix)

    # Get total images here and output progress in x for loop
    total_patches = len(x_range) * len(y_range)
    patch_num = 1
    for y in y_range:
        for x in x_range:
            print("Analyzing patch {} of {} starting from ({},{}) image coordinates...".format(patch_num, total_patches, x, y))
            image = padded_image.crop((x, y, x+look_pix, y+look_pix))
            (im_width, im_height) = image.size
            image_np = np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
            [boxes, scores, classes, num, category_index] = detect_organoids(image_np,PATH_TO_CKPT, PATH_TO_LABELS, thresh=thresh)
            for box in boxes[0]:
                cor_boxes= np.array([[y, x, y, x]]) + [box[0], box[1], box[2], box[3]]
                all_boxes= np.append(all_boxes, cor_boxes, axis=0)

            all_scores= np.append(all_scores, scores[0])
            all_classes= np.append(all_classes, classes[0])
            patch_num +=1

    all_boxes = np.array([all_boxes])
    all_scores = np.array([all_scores])
    all_classes = np.array([all_classes])

    [final_boxes, final_scores, final_classes] = nms(all_boxes, all_scores, all_classes)

    return final_boxes, final_scores, final_classes, category_index, padded_image_np, full_image_np

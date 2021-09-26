import argparse

def FBetaScoreForImage(sub, ans):
    score_over_all = 0
    num_images = len(ans)
    images_intersection = set(ans).intersection(set(sub))
    for image_intersection in images_intersection:
        score_per_image = 0
        y_true_categories = ans[image_intersection]
        num_categories = len(y_true_categories)
        y_pred_categories = sub[image_intersection]
        categories_intersection = set(y_true_categories).intersection(set(y_pred_categories))
        score_per_category = 0
        for category in categories_intersection:
            y_pred = y_pred_categories[category]
            y_true = y_true_categories[category]
            score_per_category += _compute_fbeta_pix(y_pred, y_true, 0.5)
        score_per_image += score_per_category
        score_per_image /= num_categories
        score_over_all += score_per_image
    return score_over_all / num_images

def _compute_fbeta_pix(y_pred, y_true, beta):
    """
    y_pred, y_true: dict
    {x_1:segments_1, x_2:segments_2,...}
    """
    area_true = _compute_area(y_true)
    area_pred = _compute_area(y_pred)
    x_intersection = set(y_true).intersection(set(y_pred))
    area_intersection = 0
    for x in x_intersection:
        segments_true = y_true[x]
        segments_pred = y_pred[x]
        for segment_true in segments_true:
            max_segment_true = max(segment_true)
            min_segment_true = min(segment_true)
            del_list = []
            for segment_pred in segments_pred:
                max_segment_pred = max(segment_pred)
                min_segment_pred = min(segment_pred)
                if max_segment_true >= min_segment_pred and min_segment_true <= max_segment_pred:
                    seg_intersection = min(max_segment_true, max_segment_pred) - max(min_segment_true,
                                                                                     min_segment_pred) + 1
                    area_intersection += seg_intersection
                    if max_segment_true >= max_segment_pred and min_segment_true <= min_segment_pred:
                        del_list.append(segment_pred)
            for l in del_list:
                segments_pred.remove(l)

    precision = 0
    recall = 0
    if area_pred != 0:
        precision = area_intersection/area_pred
    if area_true != 0:
        recall = area_intersection/area_true

    if precision == 0 or recall == 0:
        score = 0
    else:
        score = ((1 + beta**2)*precision*recall)/(precision*beta**2 + recall)

    return score

def _compute_area(data):
    """
    data: dict
    {x_1,segments_1, x_2:segments_2}
    """
    area = 0
    for segments in data.values():
        for segment in segments:
            area += max(segment) - min(segment) + 1
    return area


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--true-path', default = './fin.json')
    parser.add_argument('--pred-path', default = './submit.json')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    import json

    with open(args.true_path, encoding='utf-8') as f:
        ans = json.load(f)
    with open(args.pred_path, encoding='utf-8') as f:
        sub = json.load(f)

    print(FBetaScoreForImage(sub, ans))

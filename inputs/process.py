import os
import cv2
import glob
import shutil
import argparse
import numpy as np

IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def list_images(folder: str):
    files = []
    for ext in IMG_EXTS:
        files += glob.glob(os.path.join(folder, f"*{ext}"))
    return sorted(files)

def safe_copy(src: str, dst: str, overwrite: bool = False):
    if (not overwrite) and os.path.exists(dst):
        return False
    ensure_dir(os.path.dirname(dst))
    shutil.copy2(src, dst)
    return True

def make_empty_mask_from_image(img_path: str, out_mask_path: str):
    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(f"Không đọc được ảnh: {img_path}")
    h, w = img.shape[:2]
    empty = np.zeros((h, w), dtype=np.uint8)
    ensure_dir(os.path.dirname(out_mask_path))
    cv2.imwrite(out_mask_path, empty)

def find_mask_for_image(mask_dir: str, img_id: str):
    """
    Tìm mask tương ứng trong anomaly_mask theo nhiều pattern phổ biến.
    Trả về đường dẫn mask nếu tìm thấy, else None.
    """
    candidates = [
        img_id,
        f"{img_id}_mask",
        f"{img_id}-mask",
        f"{img_id}mask",
        f"{img_id}_anomaly",
        f"{img_id}-anomaly",
    ]

    # Ưu tiên match đúng tên trước
    for base in candidates:
        for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
            p = os.path.join(mask_dir, base + ext)
            if os.path.exists(p):
                return p

    # Nếu không thấy, thử tìm file bắt đầu bằng img_id (tránh bỏ sót)
    hits = []
    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        hits += glob.glob(os.path.join(mask_dir, img_id + "*" + ext))
    hits = sorted(hits)
    if len(hits) > 0:
        return hits[0]

    return None

def unique_name(dst_dir: str, filename: str):
    """Nếu trùng tên ở đích, thêm hậu tố _1,_2,..."""
    base, ext = os.path.splitext(filename)
    out = os.path.join(dst_dir, filename)
    if not os.path.exists(out):
        return out

    k = 1
    while True:
        cand = os.path.join(dst_dir, f"{base}_{k}{ext}")
        if not os.path.exists(cand):
            return cand
        k += 1

def process_split(root: str, split: str, good_name: str, bad_name: str,
                  img_subdir: str, mask_subdir: str, overwrite: bool):
    """
    root/
      split/
        good/img, good/anomaly_mask
        Ungood/img, Ungood/anomaly_mask
    => split/imgs, split/masks/0
    """
    out_img_dir = os.path.join(root, split, "imgs")
    out_mask_dir0 = os.path.join(root, split, "masks", "0")
    ensure_dir(out_img_dir)
    ensure_dir(out_mask_dir0)

    total_imgs = 0
    created_empty = 0
    copied_masks = 0
    missing_masks = 0
    renamed = 0

    for cls_name in [good_name, bad_name]:
        src_img_dir = os.path.join(root, split, cls_name, img_subdir)
        src_mask_dir = os.path.join(root, split, cls_name, mask_subdir)

        if not os.path.isdir(src_img_dir):
            print(f"[WARN] Không thấy thư mục ảnh: {src_img_dir} -> bỏ qua lớp {cls_name} ở split {split}")
            continue

        imgs = list_images(src_img_dir)
        print(f"[INFO] {split}/{cls_name}: {len(imgs)} images")

        for img_path in imgs:
            total_imgs += 1
            img_filename = os.path.basename(img_path)
            img_id, img_ext = os.path.splitext(img_filename)

            # Copy ảnh -> split/imgs/
            dst_img_path = unique_name(out_img_dir, img_filename)
            if os.path.basename(dst_img_path) != img_filename:
                renamed += 1
                # Nếu bị đổi tên ảnh vì trùng, img_id cũng phải đổi theo
                img_id = os.path.splitext(os.path.basename(dst_img_path))[0]

            safe_copy(img_path, dst_img_path, overwrite=overwrite)

            # Tìm/copy mask -> split/masks/0/<img_id>.png
            dst_mask_path = os.path.join(out_mask_dir0, img_id + ".png")

            found_mask = None
            if os.path.isdir(src_mask_dir):
                found_mask = find_mask_for_image(src_mask_dir, os.path.splitext(os.path.basename(img_path))[0])

            if found_mask is not None:
                # Copy mask (đổi tên thành <img_id>.png nếu cần)
                m = cv2.imread(found_mask, cv2.IMREAD_GRAYSCALE)
                if m is None:
                    print(f"[WARN] Mask đọc bị lỗi, tạo mask rỗng: {found_mask}")
                    make_empty_mask_from_image(dst_img_path, dst_mask_path)
                    created_empty += 1
                else:
                    # Lưu lại thành PNG chuẩn nhị phân (giữ nguyên giá trị)
                    ensure_dir(os.path.dirname(dst_mask_path))
                    cv2.imwrite(dst_mask_path, m)
                    copied_masks += 1
            else:
                # Không có mask -> tạo mask rỗng
                missing_masks += 1
                make_empty_mask_from_image(dst_img_path, dst_mask_path)
                created_empty += 1

    print(f"\n[SUMMARY] Split={split}")
    print(f"  Total images processed : {total_imgs}")
    print(f"  Masks copied          : {copied_masks}")
    print(f"  Empty masks created   : {created_empty} (missing masks: {missing_masks})")
    print(f"  Renamed due to clash  : {renamed}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True,
                        help="Dataset root, ví dụ: Breast_AD")
    parser.add_argument("--splits", type=str, default="train,valid",
                        help="Các split cần xử lý, ví dụ: train,valid,test")
    parser.add_argument("--good_name", type=str, default="good")
    parser.add_argument("--bad_name", type=str, default="Ungood")
    parser.add_argument("--img_subdir", type=str, default="img")
    parser.add_argument("--mask_subdir", type=str, default="anomaly_mask")
    parser.add_argument("--overwrite", action="store_true",
                        help="Ghi đè nếu file đích đã tồn tại")
    args = parser.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    for sp in splits:
        process_split(args.root, sp, args.good_name, args.bad_name,
                      args.img_subdir, args.mask_subdir, args.overwrite)

    print("✅ Done. Bạn sẽ có:")
    print("  <root>/train/imgs , <root>/train/masks/0")
    print("  <root>/valid/imgs , <root>/valid/masks/0")
    print("  (và split khác nếu bạn truyền thêm)")

if __name__ == "__main__":
    main()


# import os
# import glob
# import argparse
# import cv2
# import numpy as np
# from tqdm import tqdm

# IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# def ensure_dir(p: str):
#     os.makedirs(p, exist_ok=True)

# def list_img_paths(img_dir: str):
#     paths = []
#     for ext in IMG_EXTS:
#         paths += glob.glob(os.path.join(img_dir, f"*{ext}"))
#     return sorted(paths)

# def read_image(path: str):
#     img = cv2.imread(path, cv2.IMREAD_COLOR)
#     if img is None:
#         raise RuntimeError(f"Không đọc được ảnh: {path}")
#     return img

# def read_mask(path: str):
#     m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     return m  # có thể None nếu thiếu

# def make_empty_mask(h: int, w: int):
#     return np.zeros((h, w), dtype=np.uint8)

# def binarize_mask(m: np.ndarray):
#     # đưa về 0/255 (giữ đúng nhị phân)
#     m = (m > 0).astype(np.uint8) * 255
#     return m

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--root", required=True, type=str, help="Dataset root, ví dụ Breast_AD")
#     parser.add_argument("--out_root", required=True, type=str, help="Folder output, ví dụ Breast_AD_256")
#     parser.add_argument("--splits", default="train,valid", type=str, help="train,valid hoặc train,valid,test")
#     parser.add_argument("--size", default="256,256", type=str, help="target H,W. Ví dụ 256,256")
#     parser.add_argument("--img_dirname", default="imgs", type=str, help="tên folder ảnh trong mỗi split")
#     parser.add_argument("--mask_subpath", default=os.path.join("masks", "0"), type=str,
#                         help="đường dẫn tương đối tới mask trong mỗi split, mặc định masks/0")
#     parser.add_argument("--overwrite", action="store_true", help="ghi đè nếu đã tồn tại")
#     args = parser.parse_args()

#     H, W = [int(x) for x in args.size.split(",")]
#     splits = [s.strip() for s in args.splits.split(",") if s.strip()]

#     print(f"Target size: H={H}, W={W}")
#     print(f"Splits: {splits}")

#     total = 0
#     created_empty = 0
#     resized = 0
#     mismatched_pairs = 0

#     for sp in splits:
#         in_img_dir = os.path.join(args.root, sp, args.img_dirname)
#         in_mask_dir = os.path.join(args.root, sp, args.mask_subpath)

#         out_img_dir = os.path.join(args.out_root, sp, args.img_dirname)
#         out_mask_dir = os.path.join(args.out_root, sp, args.mask_subpath)
#         ensure_dir(out_img_dir)
#         ensure_dir(out_mask_dir)

#         img_paths = list_img_paths(in_img_dir)
#         print(f"\n[{sp}] Found {len(img_paths)} images in {in_img_dir}")

#         for img_path in tqdm(img_paths, desc=f"Resizing {sp}"):
#             total += 1
#             img_name = os.path.basename(img_path)
#             img_id = os.path.splitext(img_name)[0]

#             # mask path: masks/0/<img_id>.png (output mask luôn .png)
#             # input mask có thể .png/.jpg..., nên ta thử tìm theo nhiều ext
#             mask_path = None
#             for ext in IMG_EXTS:
#                 cand = os.path.join(in_mask_dir, img_id + ext)
#                 if os.path.exists(cand):
#                     mask_path = cand
#                     break

#             img = read_image(img_path)
#             mh, mw = img.shape[:2]

#             mask = None
#             if mask_path is not None:
#                 mask = read_mask(mask_path)

#             # nếu thiếu mask -> tạo mask rỗng cùng size ảnh gốc
#             if mask is None:
#                 mask = make_empty_mask(mh, mw)
#                 created_empty += 1

#             # check kích thước cặp (theo bạn là luôn đúng; nếu không đúng sẽ báo)
#             if mask.shape[0] != mh or mask.shape[1] != mw:
#                 mismatched_pairs += 1
#                 # ép mask về đúng size ảnh trước khi resize target
#                 mask = cv2.resize(mask, (mw, mh), interpolation=cv2.INTER_NEAREST)

#             # resize về target
#             img_r = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
#             mask_r = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
#             mask_r = binarize_mask(mask_r)

#             # output paths (giữ ảnh là .png để thống nhất)
#             out_img_path = os.path.join(out_img_dir, img_id + ".png")
#             out_mask_path = os.path.join(out_mask_dir, img_id + ".png")

#             if (not args.overwrite) and (os.path.exists(out_img_path) or os.path.exists(out_mask_path)):
#                 continue

#             cv2.imwrite(out_img_path, img_r)
#             cv2.imwrite(out_mask_path, mask_r)
#             resized += 1

#     print("\n===== SUMMARY =====")
#     print(f"Total samples seen        : {total}")
#     print(f"Written (resized)         : {resized}")
#     print(f"Empty masks created       : {created_empty}")
#     print(f"Pairs mismatched fixed    : {mismatched_pairs}")
#     print(f"Output written to         : {args.out_root}")
#     print("===================")

# if __name__ == "__main__":
#     main()

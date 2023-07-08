;;; tiktoken.el --- Count OpenAI Tokens -*- lexical-binding: t; -*-

;; Author: Zachary Romero
;; URL: https://github.com/zkry/text-sage
;; Version: 0.0.1
;; Package-Requires: ((emacs "28.0"))

;; This package is free software; you can redistribute it and/or modify
;; it under the terms of the GNU General Public License as published by
;; the Free Software Foundation; either version 3, or (at your option)
;; any later version.

;; This package is distributed in the hope that it will be useful,
;; but WITHOUT ANY WARRANTY; without even the implied warranty of
;; MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
;; GNU General Public License for more details.

;; You should have received a copy of the GNU General Public License
;; along with GNU Emacs.  If not, see <https://www.gnu.org/licenses/>.

;;; Commentary:
(seq-subseq [1 2 3 4 5] 0 1)


;;; Code:

(defun tiktoken-load-bpe (url)
  (let ((ht (make-hash-table :test 'equal)))
    (with-temp-buffer
      (let ((resp (request url :sync t)))
        (insert (request-response-data resp))
        (goto-char (point-min))
        (while (not (eobp))
          (let ((start (point)))
            (search-forward " ")
            (let ((str (base64-decode-string (buffer-substring-no-properties start (1- (point)))))
                  (val (string-to-number (buffer-substring-no-properties (point) (pos-eol)))))
              (puthash str val ht)
              (forward-line 1))))))
    ht))

(defun tiktoken-byte-pair-merge (piece ranks f)
  ""
  (let* ((parts (seq-into (seq-map-indexed (lambda (_ i) (vector i most-positive-fixnum))
                                           (make-vector (length piece) nil))
                          'vector))
         (get-rank (lambda (start-idx skip)
                     (if (< (+ start-idx skip 2) (length parts))
                         (let* ((b (seq-subseq piece
                                               (aref (aref parts start-idx) 0)
                                               (aref (aref parts (+ start-idx skip 2)) 0)))
                                (rank (gethash ranks (concat b))))
                           (or rank -1))
                       -1))))
    (cl-loop for i from 0 below (- (length parts) 2) do
             (let ((rank (funcall #'get-rank i 0)))
               (when (>= rank 0)
                 (setf (aref (aref parts i) 1) rank))))
    (catch 'done
     (while (> (length parts) 1)
       (let* ((min-rank most-positive-fixnum)
              (min-idx -1))
         (cl-loop for i from 0 below (- (length parts) 1) do
                  (when (< (aref (aref parts i) 1) min-rank)
                    (setq min-rank (aref (aref parts i) 1))
                    (setq min-idx i)))
         (if (< min-rank most-positive-fixnum)
             (let* ((i min-idx)
                    (rank (funcall #'get-rank i 1)))
               (if (>= rank 0)
                   (setf (aref (aref parts i) 1) rank)
                 (setf (aref (aref parts i) 1) most-positive-fixnum))
               (when (> i 0)
                 (let ((rnk (funcall #'get-rank (1- i) 1)))
                   (if (>= rk 0)
                       (setf (aref (aref parts (1- i)) 1) rk)
                     (setf (aref (aref parts (1- i)) 1) most-positive-fixnum))))
               (setq parts (seq-concatenate 'vector
                                            (seq-subseq parts 0 (1+ i))
                                            (seq-subseq parts (+ i 2)))))
           (throw 'done nil)))))
    (let ((out (make-vector (1- (length parts)) nil)))
      (cl-loop for i from 0 below (length out) do
               (setf (aref out i) (funcall f (aref (aref parts i) 0)
                                           (aref (aref parts (1+ i)) 0))))
      out)))

(defun byte-pair-encode (piece ranks)
  (if (eq (length piece) 1)
      (vector (gethash (concat piece) ranks))
    (byte-pair-merge
     piece
     ranks
     (lambda (start end)
       (gethash (concat (seq-subseq piece start end)) ranks)))))

(defvar cl100k (tiktoken-load-bpe "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"))


(defun tiktoken-tokenize (bpe text)
  )


(provide 'tiktoken)
;;; tiktoken.el ends here

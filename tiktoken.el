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


;;; Code:

(require 'ht)


;;; Encoding

(defconst tiktoken-special-endoftext "<|endoftext|>")
(defconst tiktoken-special-fim-prefix "<|fim_prefix|>")
(defconst tiktoken-special-fim-middle "<|fim_middle|>")
(defconst tiktoken-special-fim-suffix "<|fim_suffix|>")
(defconst tiktoken-special-endofprompt "<|endofprompt|>")

(defconst tiktoken-model-cl100k-base "cl100k_base") ; MODEL_CL100K_BASE
(defconst tiktoken-model-p50k-base "p50k_base") ; MODEL_P50K_BASE
(defconst tiktoken-model-p50k-edit "p50k_edit") ; MODEL_P50K_EDIT
(defconst tiktoken-model-r50k-base "r50k_base") ; MODEL_R50K_BASE

(defconst tiktoken-model-to-encoding
  (ht ("gpt-3.5-turbo" tiktoken-model-cl100k-base)
	  ("text-davinci-003" tiktoken-model-p50k-base)
	  ("text-davinci-002" tiktoken-model-p50k-base)
	  ("text-davinci-001" tiktoken-model-r50k-base)
	  ("text-curie-001" tiktoken-model-r50k-base)
	  ("text-babbage-001" tiktoken-model-r50k-base)
	  ("text-ada-001" tiktoken-model-r50k-base)
	  ("davinci" tiktoken-model-r50k-base)
	  ("curie" tiktoken-model-r50k-base)
	  ("babbage" tiktoken-model-r50k-base)
	  ("ada" tiktoken-model-r50k-base)
      ("code-davinci-002" tiktoken-model-p50k-base)
	  ("code-davinci-001" tiktoken-model-p50k-base)
	  ("code-cushman-002" tiktoken-model-p50k-base)
	  ("code-cushman-001" tiktoken-model-p50k-base)
	  ("davinci-codex" tiktoken-model-p50k-base)
	  ("cushman-codex" tiktoken-model-p50k-base)
	  ("text-davinci-edit-001" tiktoken-model-p50k-edit)
	  ("code-davinci-edit-001" tiktoken-model-p50k-edit)
	  ("text-embedding-ada-002" tiktoken-model-cl100k-base)
	  ("text-similarity-davinci-001" tiktoken-model-r50k-base)
	  ("text-similarity-curie-001" tiktoken-model-r50k-base)
	  ("text-similarity-babbage-001" tiktoken-model-r50k-base)
	  ("text-similarity-ada-001" tiktoken-model-r50k-base)
	  ("text-search-davinci-doc-001" tiktoken-model-r50k-base)
	  ("text-search-curie-doc-001" tiktoken-model-r50k-base)
	  ("text-search-babbage-doc-001" tiktoken-model-r50k-base)
	  ("text-search-ada-doc-001" tiktoken-model-r50k-base)
	  ("code-search-babbage-code-001" tiktoken-model-r50k-base)
	  ("code-search-ada-code-001" tiktoken-model-r50k-base)))

(defconst tiktoken-model-prefix-to-encoding
  (ht ("gpt-4-" tiktoken-model-cl100k-base)
      ("gpt-3.5-turbo-" tiktoken-model-cl100k-base)))

(defmemoize tiktoken-load-bpe (url)
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

(cl-defstruct (tiktoken-encoding
               (:constructor tiktoken-encoding-create)
               (:copier nil))
  name
  pat-str
  mergeable-ranks
  special-tokens)

(defun tiktoken-cl100k_base ()
  "Load ranks for cl100k_base and return it's encoder object."
  (let ((ranks (tiktoken-load-bpe "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"))
        (special-tokens (ht (tiktoken-special-endoftext 100257)
                            (tiktoken-special-fim-prefix 100258)
                            (tiktoken-special-fim-middle 100259)
                            (tiktoken-special-fim-suffix 100260)
                            (tiktoken-special-endofprompt 100276))))
    (tiktoken-encoding-create
     :name tiktoken-model-cl100k-base
     :pat-str (rx (or "'s" "'t" "'re" "'ve" "'m" "'ll" "'d"
                      (seq (? (regex "[^\r\n[:alnum:]]"))
                           (+ letter))
                      (seq (repeat 1 3 digit))
                      (seq (? " ")
                           (+ (regex "[^[:blank:][:alnum:]]"))
                           (* (in "\r\n")))
                      (seq (* (in blank))
                           (+ (in "\r\n")))
                      (seq (+ (in blank)))))
     :mergeable-ranks ranks
     :special-tokens special-tokens)))

(defun tiktoken-byte-pair-merge (piece ranks f)
  ""
  (let* ((parts (seq-into (seq-map-indexed (lambda (_ i) (vector i most-positive-fixnum))
                                           (make-vector (1+ (length piece)) nil))
                          'vector))
         (get-rank (lambda (start-idx skip)
                     (if (< (+ start-idx skip 2) (length parts))
                         (let* ((b (seq-subseq piece
                                               (aref (aref parts start-idx) 0)
                                               (aref (aref parts (+ start-idx skip 2)) 0)))
                                (rank (gethash (concat b) ranks)))
                           (or rank -1))
                       -1))))
    (cl-loop for i from 0 below (- (length parts) 2) do
             (let ((rank (funcall get-rank i 0)))
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
                    (rank (funcall get-rank i 1)))
               (if (>= rank 0)
                   (setf (aref (aref parts i) 1) rank)
                 (setf (aref (aref parts i) 1) most-positive-fixnum))
               (when (> i 0)
                 (let ((rk (funcall get-rank (1- i) 1)))
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
      (seq-into out 'list))))

(defun byte-pair-encode (piece ranks)
  (if (eq (length piece) 1)
      (vector (gethash (concat piece) ranks))
    (tiktoken-byte-pair-merge
     piece
     ranks
     (lambda (start end)
       (gethash (concat (seq-subseq piece start end)) ranks)))))

(defun tiktoken-find-regex->string-index (str regexp)
  "Find match of REGEXP in STR, returning start and end indecies."
  (save-match-data
    (let ((idx (string-match regexp str)))
      (when idx
        (cons idx (+ (length (match-string 0 str))))))))

(defun tiktoken--find-all-regexp-matches (text regexp)
  "Return all matches of REGEXP in text."
  (let ((matches))
    (with-temp-buffer
      (insert text)
      (goto-char (point-min))
      (while (search-forward-regexp regexp nil t)
        (push (match-string 0) matches)))
    (nreverse matches)))

(defun tiktoken-encode-native (encoding text allowed-special)
  ""
  (let* ((special-tokens (tiktoken-encoding-special-tokens encoding))
         (special-regex (regexp-opt (hash-table-keys special-tokens)))
         (regex (tiktoken-encoding-pat-str encoding))
         (ranks (tiktoken-encoding-mergeable-ranks encoding))
         (ret '())
         (last-piece-token-len 0)
         (start 0))
    (catch 'break2
     (while t
       (let ((next-special nil)
             (start-find start))
         (catch 'break1
           (while t
             ;; Find the next allowed special token, if any
             (let ((temp (substring text start-find (length text))))
               (setq next-special (tiktoken-find-regex->string-index temp special-regex))
               (if next-special
                   (let ((token (substring text (+ start-find (car next-special)) (+ start-find (cdr next-special)))))
                     (when (gethash token allowed-special)
                       (throw 'break1 nil))
                     (cl-incf start-find (cdr next-special)))
                 (throw 'break1 nil)))))
         (let* ((end (if next-special (+ start (car next-special)) (length text)))
                (matches (tiktoken--find-all-regexp-matches (substring text start end) regex)))
           (dolist (piece matches)
             (if-let ((token (gethash piece ranks)))
                 (progn
                   (setq last-piece-token-len 1)
                   (setq ret (append ret (list token))))
               (let ((tokens (byte-pair-encode (string-as-unibyte piece) ranks)))
                 (setq last-piece-token-len (length tokens))
                 (setq ret (append ret tokens)))))
           (if next-special
               (let* ((temp (substr text
                                    (+ start (car next-special))
                                    (+ start (cdr next-special))))
                      (token (gethash temp special-tokens))
                      (setq ret (append ret (list token)))
                      (cl-incf start (cdr next-special))
                      (setq last-piece-token-len 0)))
             (throw 'break2 nil))))))
    ret))

;; (defconst cl100k_base (tiktoken-cl100k_base))
;; (tiktoken-encode-native cl100k_base "привет!"
;;                         (tiktoken-encoding-special-tokens cl100k_base))
;; (gethash "ar" (tiktoken-encoding-mergeable-ranks cl100k_base))

(provide 'tiktoken)
;;; tiktoken.el ends here

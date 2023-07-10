;;; tiktoken.el --- Count OpenAI Tokens -*- lexical-binding: t; -*-

;; Author: Zachary Romero
;; URL: https://github.com/zkry/text-sage
;; Version: 0.0.1
;; Package-Requires: ((emacs "28.0") (ht "2.3"))
;;

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

;; fun

;;; Code:

(require 'ht)

(defgroup tiktoken nil
  "Byte-pair encoding tokenization for NLP applications."
  :group 'tools
  :link '(url-link :tag "GitHub" "https://github.com/zkry/tiktoken.el"))

(defcustom tiktoken-cache-dir
  user-emacs-directory
  "Directory to save downloaded encoding ranks.

If set to nil or an empty string, caching will be disabled."
  :group 'tiktoken
  :type 'directory)

(defcustom tiktoken-offline-ranks
  nil
  "Alist indicating the file to load for the ranks of a particular model."
  :group 'tiktoken
  :type '(alist :key-type string :value-type file))

(defconst tiktoken-special-endoftext "<|endoftext|>")
(defconst tiktoken-special-fim-prefix "<|fim_prefix|>")
(defconst tiktoken-special-fim-middle "<|fim_middle|>")
(defconst tiktoken-special-fim-suffix "<|fim_suffix|>")
(defconst tiktoken-special-endofprompt "<|endofprompt|>")

(defconst tiktoken-model-cl100k-base "cl100k_base") ; MODEL_CL100K_BASE
(defconst tiktoken-model-p50k-base "p50k_base") ; MODEL_P50K_BASE
(defconst tiktoken-model-p50k-edit "p50k_edit") ; MODEL_P50K_EDIT
(defconst tiktoken-model-r50k-base "r50k_base") ; MODEL_R50K_BASE

(defconst tiktoken-model-urls
  (ht
   (tiktoken-model-cl100k-base
    "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken")
   (tiktoken-model-p50k-edit
    "https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken")
   (tiktoken-model-p50k-base
    "https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken")
   (tiktoken-model-r50k-base
    "https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken"))
  "Mapping from model name to URL from wich to obtain the token rankings.")

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
	  ("code-search-ada-code-001" tiktoken-model-r50k-base))
  "Map of model name to encoder.")

(defconst tiktoken-model-prefix-to-encoding
  (ht ("gpt-4-" tiktoken-model-cl100k-base)
      ("gpt-3.5-turbo-" tiktoken-model-cl100k-base))
  "Map of model name prefix to encoding model.")

(defun tiktoken--parse-ranks (text)
  "Given a rank file TEXT, parse it into a map of piece to token number."
  (let* ((ht (make-hash-table :test 'equal)))
    (with-temp-buffer
      (insert text)
      (goto-char (point-min))
      (while (not (eobp))
        (let ((start (point)))
          (search-forward " ")
          (let ((str (base64-decode-string
                      (buffer-substring-no-properties start (1- (point)))))
                (val (string-to-number
                      (buffer-substring-no-properties (point) (pos-eol)))))
            (puthash str val ht)
            (forward-line 1)))))
    ht))

(defun tiktoken-load-model-bpe (model)
  "Fetch the MODEL encodings ranks and return it parsed into a hash table.

If `tiktoken-cache-dir' is non-nil and not empy, first look for
the cached file under the name
\"<tiktoken-cache-dir>/<MODEL>.txt\".  If such a file exists,
don't fetch from the external URL and use the file instead.  If
no cached file exists, fetch from the URL and save it to
mentioned filename.

If `tiktoken-offline-ranks' is an alist containing a value for
the key MODEL, then parse and use that file file in place of
fetching the URL or loading from cache."
  (let ((cache-file (and (not (s-blank-p tiktoken-cache-dir))
                         (concat tiktoken-cache-dir "/" model ".txt"))))
    (cond
     ((and cache-file (f-exists-p cache-file))
      (tiktoken--parse-ranks (f-read cache-file)))
     ((and (hash-table-p tiktoken-offline-ranks)
           (ht-get tiktoken-offline-ranks model))
      (tiktoken--parse-ranks (f-read (ht-get tiktoken-offline-ranks model))))
     (t
      (let* ((url (ht-get tiktoken-model-urls model))
             (resp (request url :sync t)))
        (unless (<= 200 (request-response-status-code resp) 299)
          (error "Unexpected result fetching model for %s at %s" model url))
        (when cache-file
          (f-write (request-response-data resp) 'utf-8 cache-file))
        (tiktoken--parse-ranks (request-response-data resp)))))))

(cl-defstruct (tiktoken-encoding
               (:constructor tiktoken-encoding-create)
               (:copier nil))
  "Structure containing all data required to byte-pair encode text."
  name
  pat-str
  mergeable-ranks
  special-tokens)

(defun tiktoken--byte-pair-merge (piece ranks f)
  "Merge bytes of PIECE according to RANKS.

F is a function taking two parameters, start and end, used to
fetch the token id from the bytes of PIECE between range start
and end."
  (let* ((parts (seq-into (seq-map-indexed
                           (lambda (_ i) (vector i most-positive-fixnum))
                           (make-vector (1+ (length piece)) nil))
                          'vector))
         (get-rank (lambda (start-idx skip)
                     (if (< (+ start-idx skip 2) (length parts))
                         (let* ((b (seq-subseq
                                    piece
                                    (aref (aref parts start-idx) 0)
                                    (aref (aref parts (+ start-idx skip 2)) 0)))
                                (rank (ht-get ranks (concat b))))
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
               (setf (aref out i)
                     (funcall f
                              (aref (aref parts i) 0)
                              (aref (aref parts (1+ i)) 0))))
      (seq-into out 'list))))

(defun tiktoken--byte-pair-encode (piece ranks)
  "Return list of token ids of PIECE split by RANKS.

RANKS is a mapping of unibyte strings to token id."
  (if (eq (length piece) 1)
      (vector (ht-get ranks (concat piece)))
    (tiktoken--byte-pair-merge
     piece
     ranks
     (lambda (start end)
       (ht-get ranks (concat (seq-subseq piece start end)))))))

(defun tiktoken-find-regex->string-index (str regexp)
  "Find match of REGEXP in STR, returning start and end indecies."
  (save-match-data
    (let ((idx (string-match regexp str)))
      (when idx
        (cons idx (+ (length (match-string 0 str))))))))

(defun tiktoken--find-all-regexp-matches (text regexp)
  "Return all matches of REGEXP in TEXT."
  (let ((matches))
    (with-temp-buffer
      (insert text)
      (goto-char (point-min))
      (while (search-forward-regexp regexp nil t)
        (push (match-string 0) matches)))
    (nreverse matches)))

(defun tiktoken--encode-native (encoding text allowed-special)
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
               (setq next-special
                     (tiktoken-find-regex->string-index temp special-regex))
               (if next-special
                   (let ((token (substring text
                                           (+ start-find (car next-special))
                                           (+ start-find (cdr next-special)))))
                     (when (ht-get allowed-special token)
                       (throw 'break1 nil))
                     (cl-incf start-find (cdr next-special)))
                 (throw 'break1 nil)))))
         (let* ((end (if next-special
                         (+ start (car next-special))
                       (length text)))
                (matches (tiktoken--find-all-regexp-matches
                          (substring text start end)
                          regex)))
           (dolist (piece matches)
             (if-let ((token (ht-get ranks piece)))
                 (progn
                   (setq last-piece-token-len 1)
                   (setq ret (append ret (list token))))
               (let ((tokens (tiktoken--byte-pair-encode
                              (string-as-unibyte piece) ranks)))
                 (setq last-piece-token-len (length tokens))
                 (setq ret (append ret tokens)))))
           (if next-special
               (let* ((temp (substr text
                                    (+ start (car next-special))
                                    (+ start (cdr next-special))))
                      (token (ht-get special-tokens temp))
                      (setq ret (append ret (list token)))
                      (cl-incf start (cdr next-special))
                      (setq last-piece-token-len 0)))
             (throw 'break2 nil))))))
    ret))

(defun tiktoken-encode (encoding text allowed-special)
  "Use ENCODING to byte-pair encode TEXT.

If ALLOWED-SPECIAL is the symbol 'all, utilize all special tokens
defined in ENCODING  If ALLOWED-SPECIAL is nil, do not allow any
special tokens.  Otherwise, ALLOWED-SPECIAL should be a list of
special tokens to use."
  (let ((allowed-special-ht
         (cond
          ((eql 'all allowed-special)
           (tiktoken-encoding-special-tokens encoding))
          ((null allowed-special) (ht))
          ((listp allowed-special)
           (let ((ht (ht)))
             (dolist (spec allowed-special)
               (ht-set ht spec t)))))))
    (tiktoken--encode-native encoding text allowed-special-ht)))

(defun tiktoken-encode-ordinary (encoding text)
  "Use ENCODING to byte-pair encode TEXT.

No special tokens are taken into account."
  (let* ((regex (tiktoken-encoding-pat-str encoding))
         (ranks (tiktoken-encoding-mergeable-ranks encoding))
         (ret '()))
    (let* ((matches (tiktoken--find-all-regexp-matches text regex)))
      (dolist (piece matches)
        (if-let ((token (ht-get ranks piece)))
            (setq ret (append ret (list token)))
          (let ((tokens (tiktoken--byte-pair-encode (string-as-unibyte piece) ranks)))
            (setq ret (append ret tokens))))))
    ret))

(defun tiktoken-decode (encoding ids)
  "Decode a list of number IDS to underlying string using ENCODING."
  (let* ((ranks (tiktoken-encoding-mergeable-ranks encoding)))
    (let* ((inv-ht (ht)))
      (ht-map (lambda (k v)
                (ht-set inv-ht v k))
              ranks)
      (string-to-multibyte
       (string-join
        (seq-map (lambda (id)
                   (ht-get inv-ht id))
                 ids))))))


;;; Encoders

(defmemoize tiktoken-cl100k-base ()
  "Load ranks for cl100k_base and return it's encoder object."
  (let ((ranks (tiktoken-load-model-bpe tiktoken-model-cl100k-base))
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

(defmemoize tiktoken-p50k-edit ()
  "Load ranks for p50k_edit and return it's encoder object."
  (let ((ranks (tiktoken-load-model-bpe tiktoken-model-p50k-edit))
        (special-tokens (ht (tiktoken-special-endoftext 50256)
                            (tiktoken-special-fim-prefix 50281)
                            (tiktoken-special-fim-middle 50282)
                            (tiktoken-special-fim-suffix 50283))))
    (tiktoken-encoding-create
     :name tiktoken-model-p50k-edit
     :pat-str (rx (or "'s" "'t" "'re" "'ve" "'m" "'ll" "'d"
                      (seq (? " ") (+ letter))
                      (seq (? " ") (+ digit))
                      (seq (? " ") (+ (regex "[^[:blank:][:alnum:]]")))
                      (seq (+ blank))))
     :mergeable-ranks ranks
     :special-tokens special-tokens)))

(defmemoize tiktoken-p50k-base ()
  "Load ranks for p50k_edit and return it's encoder object."
  (let ((ranks (tiktoken-load-model-bpe tiktoken-model-p50k-base))
        (special-tokens (ht (tiktoken-special-endoftext 50256))))
    (tiktoken-encoding-create
     :name tiktoken-model-p50k-base
     :pat-str (rx (or "'s" "'t" "'re" "'ve" "'m" "'ll" "'d"
                      (seq (? " ") (+ letter))
                      (seq (? " ") (+ digit))
                      (seq (? " ") (+ (regex "[^[:blank:][:alnum:]]")))
                      (seq (+ blank))))
     :mergeable-ranks ranks
     :special-tokens special-tokens)))

(defmemoize tiktoken-r50k-base ()
  "Load ranks for p50k_edit and return it's encoder object."
  (let ((ranks (tiktoken-load-model-bpe tiktoken-model-r50k-base))
        (special-tokens (ht (tiktoken-special-endoftext 50256))))
    (tiktoken-encoding-create
     :name tiktoken-model-r50k-base
     :pat-str (rx (or "'s" "'t" "'re" "'ve" "'m" "'ll" "'d"
                      (seq (? " ") (+ letter))
                      (seq (? " ") (+ digit))
                      (seq (? " ") (+ (regex "[^[:blank:][:alnum:]]")))
                      (seq (+ blank))))
     :mergeable-ranks ranks
     :special-tokens special-tokens)))


(defun tiktoken--encoding-from-name (encoding-name)
  "Create the model of ENCODING-NAME."
  (cond
   ((equal encoding-name tiktoken-model-cl100k-base)
    (tiktoken-cl100k-base))
   ((equal encoding-name tiktoken-model-p50k-base)
    (tiktoken-p50k-base))
   ((equal encoding-name tiktoken-model-r50k-base)
    (tiktoken-r50k-base))
   ((equal encoding-name tiktoken-model-p50k-edit)
    (tiktoken-p50k-base))
   (t (error "Unrecognized encoding name: %s" encoding-name))))

(defun tiktoken-encoding-for-model (model-name)
  "Return the encoding object of MODEL-NAME."
  (if-let ((encoding-name (ht-get tiktoken-model-to-encoding model-name)))
      (tiktoken--encoding-from-name encoding-name)
    (catch 'res
      (maphash
       (lambda (k v)
         (when (string-prefix-p k model-name)
           (throw 'res (tiktoken--encoding-from-name v))))
       tiktoken-model-prefix-to-encoding)
      (error "No encoding for model %s" model-name))))

(provide 'tiktoken)
;;; tiktoken.el ends here

;;; text-sage.el --- Interface to Language Models for Text Generation -*- lexical-binding: t; -*-

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

;;
;;
;;
;;

;;; Code:

(require 'auth-source)
;;; Machinery

;;; taken from shell-maker.el
(defun text-sage-async-shell-command (callback args extract-response &optional streaming on-complete)
  "Run ARGS in a shell asynchronously and call CALLBACK with the output."
  (let* ((buffer (and shell-maker-config (shell-maker-buffer shell-maker-config)))
         (request-id (cl-random 1000000))
         (process
          (apply #'start-process
                 (append `("text-sage-llm-openai"
                           ,(format "*text-sage-openai-%d*" request-id))
                         args)))
         (preparsed)
         (remaining-text))
    (when process
      (when streaming
        (set-process-filter
         process
         (lambda (process output)
           ;; (message "===============OUTPUT:\n%s\n\n\n" output)
           (when (buffer-live-p buffer)
             (setq remaining-text (concat remaining-text output))
             (setq preparsed (shell-maker--preparse-json remaining-text))
             (message "===============PREPARSED:\n%s\n\n\n" preparsed)
             (if (car preparsed)
                 (mapc (lambda (obj)
                         (with-current-buffer buffer
                           (funcall callback (funcall extract-response obj) t)))
                       (car preparsed))
               (with-current-buffer buffer
                 (let ((curl-exit-code (shell-maker--curl-exit-status-from-error-string (cdr preparsed))))
                   (cond ((eq 0 curl-exit-code)
                          (funcall callback (cdr preparsed) t))
                         ((numberp curl-exit-code)
                          ;; (funcall error-callback (string-trim (cdr preparsed)))
                          )
                         (t
                          (funcall callback (cdr preparsed) t))))))
             (setq remaining-text (cdr preparsed))))))
      (with-current-buffer (process-buffer process)
        (erase-buffer))
      (set-process-sentinel
       process
       (lambda (process _event)
         (let ((output (with-current-buffer (process-buffer process)
                         (buffer-string)))
               (exit-status (process-exit-status process)))

           (when (functionp on-complete)
             (funcall on-complete))
           (with-current-buffer (process-buffer process)
             (if (= exit-status 0)
                 (if (string-empty-p (string-trim output))
                     (funcall callback output nil)
                   (funcall callback (funcall extract-response output) nil))))))))
    process))

(defun text-sage--filter-nil-plist-items (plist)
  "Return PLIST with nil items removed."
  (when plist
    (let ((key (car plist))
          (val (cadr plist)))
      (if val
          (cons key (cons val (text-sage--filter-nil-plist-items (cddr plist))))
        (text-sage--filter-nil-plist-items (cddr plist))))))

;;;; LLMs

(defconst text-sage-lm-requests (make-hash-table :test 'eq))
(defvar text-sage-prevent-multiple-calls t)

(cl-defgeneric text-sage-llm-call (model prompt callback &optional stop)
  "Call Language MODEL with PROMPT and STOP and call CALLBACK with res.")

(cl-defgeneric text-sage-chat-llm-p (model)
  "Return non-nil if MODEL has a chat API."
  nil)

;;; OpenAI

(defconst text-sage-llm-openai-completions-url "https://api.openai.com/v1/completions")

(defvar text-sage-openai-key (auth-source-pick-first-password :host "api.openai.com"))

(cl-defstruct (text-sage-llm-openai
               (:constructor text-sage-llm-openai-create)
               (:copier nil))
  "Configuration of OpenAI Language Model."
  model
  (max-tokens 16)
  (temperature 1.0)
  (top-p 1.0)
  logprobs
  (presence-penalty 0)
  (frequency-penalty 0)
  (best-of 1)
  logit-bias)

(cl-defmethod text-sage-llm-call ((model text-sage-llm-openai) prompt callback &optional stop)
  "Call OpenAI Language MODEL with PROMPT and STOP and call CALLBACK with res."
  ;; sync shell command


  (let ((proc (text-sage-async-shell-command
               callback
               (list "curl"
                     text-sage-llm-openai-completions-url
                     "--fail-with-body"
                     "--no-progress-meter" "-m" "600"
                     "-H" "Content-Type: application/json"
                     "-H" (format "Authorization: Bearer %s" text-sage-openai-key)
                     "-d" (json-encode
                           (text-sage--filter-nil-plist-items
                            (list :prompt prompt
                                  :model (text-sage-llm-openai-model model)
                                  :max_tokens (text-sage-llm-openai-max-tokens model)
                                  :temperature (text-sage-llm-openai-temperature model)
                                  :top_p (text-sage-llm-openai-top-p model)
                                  :logprobs (text-sage-llm-openai-logprobs model)
                                  :stop stop
                                  :presence_penalty (text-sage-llm-openai-presence-penalty model)
                                  :frequency_penalty (text-sage-llm-openai-frequency-penalty model)
                                  :best_of (text-sage-llm-openai-best-of model)
                                  :logit_bias (text-sage-llm-openai-logit-bias model)))))
               (lambda (body)
                 (alist-get 'text
                            (aref (alist-get 'choices
                                             (if (stringp body)
                                                 (json-parse-string body :object-type 'alist)
                                               body))
                                  0)))
               nil
               (if text-sage-prevent-multiple-calls
                   (lambda ()
                     (ignore))
                 (lambda ())))))
    (when text-sage-prevent-multiple-calls
      (puthash model proc text-sage-lm-requests))))

;;; Hugging Face

(defvar text-sage-hugging-face-key (auth-source-pick-first-password :host "api-inference.huggingface.co"))

(defconst text-sage-hugging-face-inference-base-url "https://api-inference.huggingface.co/models/")

(defun text-sage-hugging-face-inference-url (model)
  "Return the URL for MODEL."
  (concat text-sage-hugging-face-inference-base-url model))

(cl-defstruct (text-sage-llm-hugging-face
               (:constructor text-sage-llm-hugging-face-create)
               (:copier nil))
  "Configuration of Hugging Face Language Model."
  model
  top-k
  top-p
  (temperature 1.0)
  repetition-penalty
  (max-new-tokens 128)
  max-time)

(cl-defmethod text-sage-llm-call ((model text-sage-llm-hugging-face) prompt callback &optional _stop)
  "Call Hugging Face Language MODEL with PROMPT and call CALLBACK with res."
  (text-sage-async-shell-command
   callback
   (list "curl"
         (text-sage-hugging-face-inference-url (text-sage-llm-hugging-face-model model))
         "-X" "POST"
         "--fail-with-body"
         "--no-progress-meter" "-m" "600"
         "-H" (format "Authorization: Bearer %s" text-sage-hugging-face-key)
         "-d"
         (json-encode
          (list :inputs prompt
                :parameters
                (text-sage--filter-nil-plist-items
                 (list :return_full_text json-false
                       :top_k (text-sage-llm-hugging-face-top-k model)
                       :top_p (text-sage-llm-hugging-face-top-p model)
                       :temperature (text-sage-llm-hugging-face-temperature model)
                       :repetition_penalty (text-sage-llm-hugging-face-repetition-penalty model)
                       :max_new_tokens (text-sage-llm-hugging-face-max-new-tokens model)
                       :max_time (text-sage-llm-hugging-face-max-time model)))
                :options
                (list :use_cache t
                      :wait_for_model t
                      :return_full_text t
                      :wait_time 600))))
   (lambda (body)
     (gethash "generated_text" (aref (json-parse-string body) 0)))))

;;; Chat Models

(defconst text-sage-llm-openai-chat-completions-url "https://api.openai.com/v1/chat/completions")

(cl-defgeneric text-sage-llm-chat-call (model messages callback &optional stop)
  "Call Language MODEL with MESSAGES and STOP and call CALLBACK with res.")

(cl-defstruct (text-sage-llm-openai-chat
               (:constructor text-sage-llm-openai-chat-create)
               (:copier nil))
  "Configuration of OpenAI Language Model."
  model
  max-tokens
  n
  temperature
  top-p
  presence-penalty
  frequency-penalty
  logit-bias
  ;; format should JSON encode to the spec
  ;; https://platform.openai.com/docs/api-reference/chat/create#chat/create-functions
  functions
  function-call)

(cl-defmethod text-sage-chat-llm-p ((model text-sage-llm-openai-chat)) t)

(cl-defmethod text-sage-llm-chat-call ((model text-sage-llm-openai-chat) messages callback &optional stop)
  "Call OpenAI Language MODEL with MESSAGES and STOP and call CALLBACK with res."
  (when (and text-sage-prevent-multiple-calls (gethash model text-sage-lm-requests))
    (error "Model has an in progress request: %s" model))
  (let ((formatted-messages
         (seq-map (pcase-lambda (`(,role ,msg))
                    `((role . ,(substring (symbol-name role) 1))
                      (content . ,msg)))
                  messages)))
    (let ((proc (text-sage-async-shell-command
                 callback
                 (list "curl"
                       text-sage-llm-openai-chat-completions-url
                       "--fail-with-body"
                       "--no-progress-meter" "-m" "600"
                       "-H" "Content-Type: application/json"
                       "-H" (format "Authorization: Bearer %s" text-sage-openai-key)
                       "-d" (json-encode
                             (text-sage--filter-nil-plist-items
                              (list :messages formatted-messages
                                    :model (text-sage-llm-openai-chat-model model)
                                    :max_tokens (text-sage-llm-openai-chat-max-tokens model)
                                    :n (text-sage-llm-openai-chat-n model)
                                    :temperature (text-sage-llm-openai-chat-temperature model)
                                    :top_p (text-sage-llm-openai-chat-top-p model)
                                    :stop stop
                                    :presence_penalty (text-sage-llm-openai-chat-presence-penalty model)
                                    :frequency_penalty (text-sage-llm-openai-chat-frequency-penalty model)
                                    :logit_bias (text-sage-llm-openai-chat-logit-bias model)
                                    :functions (text-sage-llm-openai-chat-functions model)
                                    :function-call
                                    (text-sage-llm-openai-chat-function-call model)))))
                 (lambda (body)
                   ;; TODO refactor this.
                   (alist-get 'content
                              (alist-get 'message
                                         (aref (alist-get 'choices
                                                          (if (stringp body)
                                                              (json-parse-string body :object-type 'alist)
                                                            body))
                                               0))))
                 nil
                 (if text-sage-prevent-multiple-calls
                     (lambda ()
                       (ignore))
                   (lambda ())))))
      (when text-sage-prevent-multiple-calls
       (puthash model proc text-sage-lm-requests)))))

;;; Tokens

(defun text-sage-count-tokens (message)
  (cond
   ((stringp message)
    (/ (length message) 4))
   ((text-sage-chat-prompt-p message)
    (seq-reduce
     (pcase-lambda (acc `(,_ ,msg))
       (+ acc (/ (length msg) 4)))
     message
     0))))

;;; Prompts

(defconst text-sage-agent-labels
  '((:user . "User")
    (:system . "System")
    (:assistant . "Assistant"))
  "Labels for chat prompts to format as raw text.")

(defconst text-sage-parsed-prompt-instructions-separator "\n\n")

(cl-defstruct (text-sage-parsed-prompt
               (:constructor text-sage-parsed-prompt-create)
               (:copier nil))
  parser
  prompt)

(defun text-sage-chat-prompt-p (prompt)
  "Return non-nil if PROMPT is a chat prompt.

A chat prompt is a list of messages.  A message is a list where
the first element is a keyword (ex :system, :assistant) and the
second a string."
  (and (listp prompt)
       (seq-every-p
        (lambda (elt)
          (or (and (symbolp elt)
                   (string-prefix-p ":@" (symbol-name elt)))
              (and (listp elt)
                   (= 2 (length elt))
                   (keywordp (car elt))
                   (stringp (cadr elt)))))
        prompt)))

(defun text-sage--chat-to-text-format (prompt vars)
  "Format chat PROMPT with VARS."
  (unless (text-sage-chat-prompt-p prompt)
    (error "Prompt must be a valid chat prompt"))
  (text-sage-format
   (string-join
    (seq-map
     (pcase-lambda (`(,agent ,message))
       (format "%s: %s" (alist-get agent text-sage-agent-labels) message))
     prompt)
    "\n")
   vars))

(defun text-sage-chat-format (prompt vars)
  (let* ((format-instructions))
    (unless (or (text-sage-parsed-prompt-p prompt)
                (text-sage-chat-prompt-p prompt))
      (error "Prompt is not a chat prompt: %s" prompt))
    (when (text-sage-parsed-prompt-p prompt)
      (setq format-instructions (text-sage-parser-format-instructions
                                 (text-sage-parsed-prompt-parser prompt)))
      (setq prompt (text-sage-parsed-prompt-prompt prompt))
      (unless (text-sage-chat-prompt-p prompt)
        (error "Prompt of parser is not a chat prompt: %s" prompt)))
    (let ((formatted-prompt
           (seq-reduce
            (lambda (acc elt)
              (cond
               ((and (symbolp elt)
                     (string-prefix-p ":@" (symbol-name elt)))
                (let* ((replacement (intern (substring (symbol-name elt) 2)))
                       (replacement-elt (alist-get replacement vars)))
                  (unless (assoc replacement vars)
                    (error "no replacement variable found for variable %s" replacement))
                  (cond
                   ((text-sage-chat-prompt-p replacement-elt)
                    (append acc replacement-elt))
                   ((text-sage-chat-prompt-p (list replacement-elt))
                    (append acc (list replacement-elt)))
                   (t (error "invalid type for replacement %s" replacement)))))
               ((listp elt)
                (pcase-let* ((`(,agent ,msg) elt))
                  (append acc (list (list agent (text-sage-format msg vars))))))))
            prompt
            '())))
      (when format-instructions
        (setq formatted-prompt (append formatted-prompt `((:system ,format-instructions)))))
      formatted-prompt)))

(defun text-sage-format (prompt vars)
  "Format PROMPT with variables VARS to be ready for LLM consumption."
  (cond
   ((text-sage-parsed-prompt-p prompt)
    (let* ((sub-prompt (text-sage-parsed-prompt-prompt prompt))
           (formatted-sub-prompt (text-sage-format sub-prompt vars))
           (parser (text-sage-parsed-prompt-parser prompt))
           (parse-instructions (text-sage-parser-format-instructions parser)))
      ;; parse instructions are tacked on to the end of a plain text prompt.
      (concat formatted-sub-prompt text-sage-parsed-prompt-instructions-separator parse-instructions )))
   ((text-sage-chat-prompt-p prompt)
    (text-sage--chat-to-text-format prompt vars))
   ((stringp prompt)
    (with-temp-buffer
      (insert prompt)
      (pcase-dolist (`(,key . ,val) vars)
        (goto-char (point-min))
        (while (search-forward-regexp
                (format "{{ *%s *}}" (regexp-quote (if (symbolp key) (symbol-name key) key)))
                nil t)
          (replace-match (regexp-quote val) nil t)))
      (buffer-string)))
   (t (error "Invalid prompt %s" prompt))))

;; TODO: can I delete this?
(defun text-sage-format-prompt (prompt vars)
  (unless (text-sage-chat-prompt-p prompt)
    (error "prompt must be a valid chat prompt."))
  (seq-map
   (pcase-lambda (`(,agent ,message))
     (list agent (text-sage-format message vars)))
   prompt))

;;; Examples

(defvar text-sage-example-separator "\n\n")

(cl-defstruct (text-sage-example
               (:constructor text-sage-example-create)
               (:copier nil))
  template
  items)

(defun text-sage-example-selector-by-length (max-length &optional get-text-length)
  (lambda (example)
    (unless (text-sage-example-p example)
      (error "invalid type of example"))
    (let ((template (text-sage-example-template example))
          (items (text-sage-example-items example))
          (text ""))
      (catch 'done
        (while items
          (let* ((at-item (car items))
                 (next-text (concat text (if (equal text "")
                                             ""
                                           text-sage-example-separator)
                                    (text-sage-format template at-item))))
            (if (<= (funcall (or get-text-length #'length) next-text) max-length)
                (setq text next-text)
              (throw 'done nil)))
          (setq items (cdr items))))
      text)))


;;; Documents and Text Splitters

(cl-defstruct (text-sage-document
               (:constructor text-sage-document-create)
               (:copier nil))
  content
  metadata)

(cl-defstruct (text-sage-character-text-splitter
               (:constructor text-sage-character-text-splitter-create)
               (:copier nil))
  (separator "\n\n")
  (chunk-size 4000)
  (chunk-overlap 200)
  (length-function #'length))


(cl-defgeneric text-sage-text-split (text-splitter text)
  "Split TEXT using TEXT-SPLITTER.")

(defun text-sage--merge-splits (splits length-function chunk-size separator chunk-overlap)
  "Combine SPLITS into documents according to sizing rules.

Each produced document should not be longer than CHUNK-SIZE.
Each document will have an overlap of the preceding section at
most of size CHUNK-OVERoLAP.  The pieces that are joined together
in a document will be separateb by SEPARATOR.  Lengths are
determined by LENGTH-FUNCTION."
  (let* ((documents '()))
    (let* ((i 0)
           (separator-len (funcall length-function separator)))
      (catch 'outer-loop
       (while t
         (let* ((current-document (nth i splits))
                (current-size (funcall length-function current-document)))
           (cl-incf i)
           (when (and (numberp chunk-overlap) (> chunk-overlap 0))
             (let* ((j (- i 2))
                    (overlap (nth j splits))
                    (overlap-size (funcall length-function overlap)))
               (when (and (>= j 0)
                          (<= overlap-size chunk-overlap)
                          (<= (+ overlap-size separator-len current-size) chunk-size))
                 (cl-decf j)
                 (catch 'overlap-loop
                   (while t
                     (let* ((prev-elt (nth j splits))
                            (prev-elt-len (length prev-elt))
                            (past-start (< j 0)))
                       (cond
                        ((or past-start
                             (> (+ overlap-size separator-len prev-elt-len) chunk-overlap)
                             (> (+ current-size separator-len overlap-size prev-elt-len) chunk-size))
                         (throw 'overlap-loop nil))
                        (t
                         (setq overlap (concat prev-elt separator overlap))
                         (setq overlap-size (+ overlap-size separator-len prev-elt-len))))
                       (cl-decf j))))
                 (setq current-document (concat overlap separator current-document))
                 (setq current-size (+ current-size separator-len overlap-size)))))
           (catch 'doc-loop
             (while t
               (let* ((next-elt (nth i splits))
                      (next-elt-len (length next-elt)) ;; TODO: don't call length
                      (at-end (= i (length splits))))
                 (cond
                  (at-end
                   (push current-document documents)
                   (throw 'outer-loop nil))
                  ((> (+ current-size separator-len next-elt-len) chunk-size)
                   (push current-document documents)
                   (throw 'doc-loop nil))
                  (t
                   (setq current-document (concat current-document separator next-elt))
                   (setq current-size (+ current-size separator-len next-elt-len))))
                 (cl-incf i))))))))
    (nreverse documents)))

(cl-defmethod text-sage-text-split ((splitter text-sage-character-text-splitter) text)
  ""
  (let* ((separator (text-sage-character-text-splitter-separator splitter))
         (length-function (text-sage-character-text-splitter-length-function splitter))
         (chunk-size (text-sage-character-text-splitter-chunk-size splitter))
         (chunk-overlap (text-sage-character-text-splitter-chunk-overlap splitter))
         (splits (string-split text separator)))
    (text-sage--merge-splits  splits length-function chunk-size separator chunk-overlap)))

(cl-defstruct (text-sage-recursive-character-splitter
               (:constructor text-sage-recursive-character-splitter-create)
               (:copier nil))
  (separators '("\n\n" "\n" " " ""))
  (chunk-size 4000)
  (length-function #'length))

(defun text-sage--split-text-evenly (text size length-function)
  "Divide TEXT into evenly sized chunks no greater than SIZE."
  ;; TODO: use a function
  (let* ((part (ceiling (/ (float (funcall length-function text)) size)))
         (part-size-bytes (ceiling (/ (funcall length-function text) part))))
    (seq-map (lambda (chars)
               (apply #'string chars))
             (-partition-in-steps part-size-bytes part-size-bytes (seq-into text 'list)))))

(cl-defmethod text-sage-text-split ((splitter text-sage-recursive-character-splitter) text &optional separators)
  "Split TEXT according to the recursive character SPLITTER."
  (let* ((separators (or separators (text-sage-recursive-character-splitter-separators splitter)))
         (length-function (text-sage-recursive-character-splitter-length-function splitter))
         (chunk-size (text-sage-recursive-character-splitter-chunk-size splitter))
         (chunk-overlap (text-sage-recursive-character-splitter-chunk-overlap splitter))
         (top-separator (car separators))
         (splits (if (equal top-separator "")
                     (text-sage--split-text-evenly text chunk-size length-function)
                   (string-split text top-separator))))
    (seq-mapcat
     (lambda (split)
       (if (> (funcall length-function split) chunk-size)
           (text-sage-text-split splitter split (append (cdr separators) '("")))
         (list split)))
     splits)))

(defun text-sage-documents-to-inputs (documents &optional is-chat)
  "Convert a list of DOCUMENTS to a chain input value.

If IS-CHAT is non-nil, return a chat message."
  (let* ((context-string
          (string-join
           (seq-map (lambda (document)
                      (text-sage-document-content document))
                    documents)
           "\n\n")))
    (if is-chat
        `(:system ,(format "Context: %s" context-string))
      context-string)))

;;; Parsers

(cl-defstruct (text-sage-parser
               (:constructor text-sage-parser-create)
               (:copier nil))
  "A parser for the output of a LLM."
  format-instructions
  parse-function
  parse-with-prompt-function
  retries)

(defvar text-sage-json-parser
  (text-sage-parser-create
   :format-instructions
   "The output should be properly formatted JSON."
   :parse-function
   (lambda (str)
     (json-parse-string str))))

(defun text-sage--validate-spec (type value)
  "Return an error string if VALUE is not of TYPE.

Refer to `text-sage-spec-parser' for valid types."
  (let ((type-of
         (lambda (val)
           (cond
            ((stringp val) (format "\"%s\"" val))
            ((numberp val) val)
            ((booleanp val) (if (or (not val) (eql val :false))
                                "false"
                              "true"))
            ((json-alist-p val) (json-encode val))
            ((seqp val) (json-encode val))))))
    (pcase type
      ('string
       (when (not (stringp value))
         (format "value %s is not of type string" (funcall type-of value))))
      ('boolean
       (when (not (or (booleanp value) (eql value :false)))
         (format "value %s is not of type boolean" (funcall type-of value))))
      ('number
       (when (not (numberp value))
         (format "value %s is not of type number" (funcall type-of value))))
      (`(array ,type)
       (cond
        ((not (seqp value)) (format "value %s is not an array" (funcall type-of value)))
        (t (seq-find (lambda (elt) (text-sage--validate-spec type elt)) value))))
      ((pred
        (lambda (x) (eql (car x) 'category)))
       (when (not (member value (cdr type)))
         (format "value %s is not one of %s"
                 (funcall type-of value)
                 (string-join (seq-map (lambda (x) (format "\"%s\"" x))
                                       (cdr type))
                              ", ")))))))

(defun text-sage--format-spec-type (type)
  "Format TYPE to a string to give to the LLM."
  (pcase type
    ('string "string")
    ('number "number")
    ('boolean "boolean")
    (`(array ,x) (format "an array of %s" (text-sage--format-spec-type x)))
    ((pred
      (lambda (x) (eql (car x) 'category)))
     (format "only the string values %s" (string-join (cdr type) ", ")))))

(defconst text-sage-spec-parser-template "The output should be properly formatted JSON. You must only output JSON and no other commentary.  \
The JSON should be a single object with the the fields:
%s")

(defun text-sage-spec-parser (specification &optional field-comments example)
  "Return a JSON parser for data SPECIFICATION, with EXAMPLE and FIELD-COMMENTS.

SPECIFICATION should be an alist of (FIELD-NAME . TYPE).
EXAMPLES should be an alist of (FIELD-NAME . EXAMPLE-VALUE).  The
symbols following are valid types: string, boolean, number.

Type may also be (array TYPE), signifying an array of TYPE.

If provided, EXAMPLE should be data that when encoded with
`json-encode' produces an example like the desired output."
  (let* ((field-names (mapcar #'car specification))
         (spec-text (string-join
                     (seq-map (lambda (field-name)
                                (let ((type-sym (alist-get field-name specification nil nil #'equal))
                                      (comment (alist-get field-name field-comments nil nil #'equal)))
                                  (format "- %s: type %s%s"
                                          field-name
                                          (text-sage--format-spec-type type-sym)
                                          (if comment
                                              (format ", %s" comment)
                                            ""))))
                              field-names)
                     "\n"))
         (instructions (format text-sage-spec-parser-template spec-text))
         (instructions (if example
                           (concat instructions "\n\n The following is an example JSON object:\n"
                                   (json-encode example))
                         instructions)))
    (text-sage-parser-create
     :format-instructions
     instructions
     :parse-function
     (lambda (str)
       (let ((data (json-parse-string str :object-type 'alist)))
         (unless (json-alist-p data)
           (error "Response is not a JSON object"))
         (seq-map
          (pcase-lambda (`(,field-name . ,type))
            (let ((err (text-sage--validate-spec type (alist-get field-name data nil nil #'equal))))
              (when err
                (error "Error on field %s: %s" field-name err))))
          specification)
         data)))))

(defun text-sage-parse (parser str)
  "Parse STR with PARSER."
  (unless (text-sage-parser-p parser)
    (error "Invalid parser type: %S" parser))
  (let ((parse-func (text-sage-parser-parse-function parser)))
    (funcall parse-func str)))

(defconst text-sage-correction-prompt
  "Data was requested with the following specification:

{{spec}}

The following response was returned by an AI:

{{given}}

For this response, the following error occurred:

{{err}}

The following is a correction of the response returned by the AI:\n\n")

(defun text-sage-correct-parsing-error (llm parser given error-message callback &optional retries)
  "Query LLM to correct GIVEN according to ERROR-MESSAGE and PARSER, calling CALLBACK.

If RETRIES is a positive number, rerun the function to attempt to
get a correct parse."
  (when (not retries)
    (setq retries (text-sage-parser-retries parser)))
  (let* ((instructions (text-sage-parser-format-instructions parser))
         (prompt (text-sage-format text-sage-correction-prompt
                                   `((spec . ,instructions)
                                     (given . ,given)
                                     (err . ,error-message)))))
    (text-sage-llm-call
     llm
     prompt
     (lambda (res _)
       (condition-case err
           (let* ((parsed (text-sage-parse parser res)))
             (funcall callback parsed nil))
         (error
          (if (and retries (> retries 0))
              (text-sage-correct-parsing-error llm parser res (cadr err) callback (1- retries))
            (error "LLM returned unparsable results: %s" res))))))))

(defun text-sage-parse-with-correction (llm parser str callback)
  "Attempt to parse STR with PARSER.  If parser return error, use LLM to fix it.

The parsed result is passed as the first argument to CALLBACK."
  (condition-case err
      (funcall callback (text-sage-parse parser str) nil)
    (error
     (let* ((err-msg (cadr err)))
       (text-sage-correct-parsing-error llm parser str err-msg callback)))))


;;; Memory

(defvar text-sage-conversation-memory-human-tag "Human")
(defvar text-sage-conversation-memory-ai-tag "AI")

(cl-defstruct (text-sage-conversation-buffer-memory
               (:constructor text-sage-conversation-buffer-memory-create)
               (:copier nil))
  messages)

(cl-defgeneric text-sage-save-memory-context (memory input output))

(cl-defmethod text-sage-save-memory-context
  ((memory text-sage-conversation-buffer-memory) input output)
  (let ((messages (text-sage-conversation-buffer-memory-messages memory)))
    (setf (text-sage-conversation-buffer-memory-messages memory)
          (append messages
                  `((:user ,input)
                    (:assistant ,output))))))

(cl-defgeneric text-sage-load-memory-variables (memory))

(cl-defmethod text-sage-load-memory-variables ((memory text-sage-conversation-buffer-memory))
  (let* ((messages (text-sage-conversation-buffer-memory-messages memory))
         (formatted-message
          (string-join
           (seq-map
            (pcase-lambda (`(,agent ,msg))
              (if (equal agent :user)
                  (format "%s: %s" text-sage-conversation-memory-human-tag msg)
                (format "%s: %s" text-sage-conversation-memory-ai-tag msg)))
            messages)
           "\n")))
    `((history-messages . ,messages)
      (history . ,formatted-message))))

(cl-defstruct (text-sage-conversation-buffer-window-memory
               (:constructor text-sage-conversation-buffer-window-memory)
               (:copier nil))
  messages
  (window-size 1))

(cl-defmethod text-sage-save-memory-context
  ((memory text-sage-conversation-buffer-window-memory) input output)
  (let ((messages (text-sage-conversation-buffer-window-memory-messages memory)))
    (setf (text-sage-conversation-buffer-window-memory-messages memory)
          (append messages
                  `((:user ,input)
                    (:assistant ,output))))))

(cl-defmethod text-sage-load-memory-variables ((memory text-sage-conversation-buffer-window-memory))
  (let* ((window-size (text-sage-conversation-buffer-window-memory-window-size memory))
         (messages (text-sage-conversation-buffer-window-memory-messages memory))
         (formatted-message
          (string-join
           (seq-map
            (pcase-lambda (`(,agent ,msg))
              (if (equal agent :user)
                  (format "%s: %s" text-sage-conversation-memory-human-tag msg)
                (format "%s: %s" text-sage-conversation-memory-ai-tag msg)))
            (seq-reverse (seq-take (seq-reverse messages) window-size)))
           "\n")))
    `((history-messages . ,messages)
      (history . ,formatted-message))))

(cl-defstruct (text-sage-conversation-summary-buffer-memory
               (:constructor text-sage-conversation-summary-buffer-memory-create)
               (:copier nil))
  llm
  messages
  (max-token-limit 1000)
  (compression-rate 0.5)
  summary)

(defconst text-sage-summarizer-prompt
  '((:system "Progressively summarize the lines of conversation provided, adding onto the previous summary returning a new summary.\n
EXAMPLE
Current summary:
The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good.

New lines of conversation:
Human: Why do you think artificial intelligence is a force for good?
AI: Because artificial intelligence will help humans reach their full potential.

New summary:
The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good because it will help humans reach their full potential.
END OF EXAMPLE")
    (:user "Produce a new summary given the following:

Current summary:
{{summary}}

New lines of conversation:
{{new-lines}}")))

(cl-defmethod text-sage-save-memory-context
  ((memory text-sage-conversation-summary-buffer-memory) input output)
  ""
  (let* ((messages (append (text-sage-conversation-summary-buffer-memory-messages memory)
                           `((:user ,input)
                             (:assistant ,output))))
         (max-token-limit (text-sage-conversation-summary-buffer-memory-max-token-limit memory))
         (msg-size (text-sage-count-tokens messages)))
    (if (> msg-size max-token-limit)
        (let* ((compression-rate (text-sage-conversation-summary-buffer-memory-compression-rate memory))
               (summary (text-sage-conversation-summary-buffer-memory-summary memory))
               (llm (text-sage-conversation-summary-buffer-memory-llm memory))
               (drop-count (ceiling (* (length messages) compression-rate)))
               (to-be-summarized (seq-reverse (seq-take (seq-reverse messages) drop-count)))
               (remaining (seq-drop messages drop-count)))
          ;; TODO: This will almost certainly result in clobbered data if two requests are sent
          ;; close enough to eachother.
          (text-sage-generic-llm-call
           llm text-sage-summarizer-prompt
           `((summary . ,(or summary "<no summary>"))
             (new-lines . ,(text-sage-format to-be-summarized nil)))
           (lambda (res _)
             (setf (text-sage-conversation-summary-buffer-memory-messages memory) remaining)
             (setf (text-sage-conversation-summary-buffer-memory-summary memory) res))))
      (message "===")
      (setf (text-sage-conversation-summary-buffer-memory-messages memory) messages))))

(cl-defmethod text-sage-load-memory-variables ((memory text-sage-conversation-summary-buffer-memory))
  ""
  (let* ((messages (text-sage-conversation-summary-buffer-memory-messages memory))
         (summary (text-sage-conversation-summary-buffer-memory-summary memory))
         (formatted-message
          (format "%s%s"
                  (if summary (concat "System: %s\n\n" summary) "")
                  (string-join
                   (seq-map
                    (pcase-lambda (`(,agent ,msg))
                      (if (equal agent :user)
                          (format "%s: %s" text-sage-conversation-memory-human-tag msg)
                        (format "%s: %s" text-sage-conversation-memory-ai-tag msg)))
                    messages)
                   "\n"))))
    `((history-messages . ,(if summary (cons (list :system summary) messages) messages))
      (history . ,formatted-message))))

;;; Chains

(defvar text-sage-disable-parser nil
  "When non-nil, skip the parsing step of a chain.")

(defun text-sage-generic-llm-call (llm prompt inputs callback)
  "Based on the type of LLM and PROMPT, make an appropriate call to the LLM.

PROMPT is formatted as a chat if LLM supports chat interface.
CALLBACK is called when request finished."
  (if (text-sage-chat-llm-p llm)
      (let* ((formatted-prompt (text-sage-chat-format prompt inputs)))
        (text-sage-llm-chat-call llm formatted-prompt callback))
    (let* ((formatted-prompt (text-sage-format prompt inputs)))
      (text-sage-llm-call llm formatted-prompt callback))))

(cl-defstruct (text-sage-llm-chain
               (:constructor text-sage-llm-chain-create)
               (:copier nil))
  prompt
  llm)

(cl-defgeneric text-sage-chain-run (chain inputs callback))

(cl-defmethod text-sage-chain-run ((chain text-sage-llm-chain) inputs callback)
  ""
  (let* ((prompt (text-sage-llm-chain-prompt chain))
         (llm (text-sage-llm-chain-llm chain))
         (parser (and (text-sage-parsed-prompt-p prompt)
                      (text-sage-parsed-prompt-parser prompt)))
         (callback
          (if (or (not parser) text-sage-disable-parser)
              callback
            (lambda (result _partial)
              (text-sage-parse-with-correction llm parser result callback)))))
    (text-sage-generic-llm-call llm prompt inputs callback)))

(defvar text-sage-conversation-chain-default-prompt
  "The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{{history}}
Human: {{input}}
AI:")

(defvar text-sage-conversation-chain-default-chat-prompt
  '((:system "The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.")
    :@history-messages
    (:user "{{input}}")))

(cl-defstruct (text-sage-conversation-chain
               (:constructor text-sage-conversation-chain-create)
               (:copier nil))
  llm
  (prompt text-sage-conversation-chain-default-prompt)
  memory)

(defun text-sage--conversation-chain-update-memory (chain input output)
  (let ((memory (text-sage-conversation-chain-memory chain)))
    (text-sage-save-memory-context memory input output)))

(cl-defmethod text-sage-chain-run ((chain text-sage-conversation-chain) inputs callback)
  (when (stringp inputs)
    (setq inputs `((input . ,inputs))))
  (let* ((prompt (text-sage-conversation-chain-prompt chain))
         (llm (text-sage-conversation-chain-llm chain))
         (memory (text-sage-conversation-chain-memory chain))
         (history (text-sage-conversation-chain-memory chain))
         (inputs (append inputs (text-sage-load-memory-variables history)))
         (parser (and (text-sage-parsed-prompt-p prompt)
                      (text-sage-parsed-prompt-parser prompt)))
         (cb (if (or (not parser) text-sage-disable-parser)
                      callback
                    (lambda (result _partial)
                      (text-sage-parse-with-correction llm parser result callback)))))
    (text-sage-generic-llm-call
     llm prompt inputs
     (lambda (result partial)
       (when (not partial)
         (if (text-sage-chat-prompt-p prompt)
             (let ((last-msg (cadar (last (text-sage-chat-format prompt inputs)))))
               (text-sage--conversation-chain-update-memory chain last-msg result))
           (text-sage--conversation-chain-update-memory
            chain (alist-get 'input inputs) result)))
       (funcall cb result partial)))))


(defun text-sage-qa-stuff-chain (llm input-documents)
  (text-sage-llm-chain-create
   :llm llm
   :prompt `((:system "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.")
             ,(text-sage-documents-to-inputs input-documents t)
             (:user "{{input}}"))))

;;; Playground


;; (defconst test-linter-chain
;;   (let* ((llm (text-sage-llm-openai-create :model "text-davinci-003" :max-tokens 100)))
;;     (text-sage-llm-chain-create
;;      :prompt "You are an AI assistant designed to help programmers. The programmer asks you to look at the following Go code:

;; {{code}}

;; Human: Is there any obvious bug regarding this code? If you find something respond in the format {\"looksGood\": <true|false> \"response\": \"<your comment here>\"}.
;; AI: "
;;      :llm llm)))

;; (text-sage-chain-run test-linter-chain '((code . "func addTwoNumbers(a, b int) int {
;; 	return a - b
;; }"))
;;                      (lambda (msg _) (message ">>> %s" msg)))


;; (text-sage-chain-run test-linter-chain '((code . "func addTwoNumbers(a, b int) int {
;; 	retrun a + b
;; }"))
;;                      (lambda (msg _) (message ">>> %s" msg)))

;; (defconst testchain
;;   (let* ((llm (text-sage-llm-openai-create :model "text-davinci-003" :max-tokens 100))
;;          (memory (text-sage-conversation-buffer-memory-create))
;;          (chain (text-sage-conversation-chain-create
;;                  :memory memory
;;                  :llm llm)))
;;     chain))

;; (text-sage-conversation-chain-memory testchain)

;; (text-sage-chain-run testchain "What is your name?" (lambda (msg _) (message ">>> %s" msg)))

;; (text-sage-chain-run testchain "Soryy, I couldn't hear that. What was that again?" (lambda (msg _) (message ">>> %s" msg)))

;; (text-sage-chain-run testchain "Awesome. I'm happy to meet you." (lambda (msg _) (message ">>> %s" msg)))

;; (let* ((llm (text-sage-llm-openai-create :model "text-davinci-003" :max-tokens 100))
;;        (chain (text-sage-llm-chain-create
;;                :prompt "The following is a joke about {{topic}}:\n"
;;                :llm llm)))
;;   (text-sage-chain-run chain '((topic . "stars"))
;;                        (lambda (msg _) (message ">>> %s" msg)))
;;   (text-sage-chain-run chain '((topic . "tables"))
;;                        (lambda (msg _) (message ">>> %s" msg))))

;; (let* ((mem (text-sage-conversation-buffer-memory-create)))
;;   (text-sage-save-memory-context mem "What is the color of the sky?" "Blue.")
;;   (text-sage-load-memory-variables mem))

;; (let* ((mem (text-sage-conversation-buffer-memory-create)))
;;   (text-sage-save-memory-context mem "What is the color of the sky?" "Blue.")
;;   (text-sage-load-memory-variables mem))

;; (text-sage-format
;;  '((:system "You are a {{ adjective }} robot") (:user "sunny {{ game }}"))
;;  '((game . "baseball")
;;    (adjective . "funny")))

;; (text-sage-chat-prompt-p '((:system "You are a killer robot") (:user "sunny")))

;; (text-sage-format "{{abc}}->{{def}}" '((abc . "a") (def . "d")))

;; (text-sage-format
;;  '((:system "Your name is Martin")
;;    (:user "{question}"))
;;  '((abc . "a") (def . "d")))

;; (let ((llm (text-sage-llm-openai-create
;;             :model "text-davinci-003")))
;;   (text-sage-llm-call
;;    llm
;;    "Once upon a time, there was a"
;;    (lambda (res _partialp) (message ">>> %S" res))))

;; (let ((llm (text-sage-llm-hugging-face-create
;;             :model "stabilityai/stablelm-tuned-alpha-3b")))
;;   (text-sage-llm-call
;;    llm
;;    "You are an AI that only answers with one word. Question: True or false: the sky is blue. Response: "
;;    (lambda (res _partialp) (message "|>>> %S" res))))

;; (let ((llm (text-sage-llm-openai-chat-create
;;             :model "gpt-3.5-turbo")))
;;   (text-sage-llm-chat-call
;;    llm
;;    '((("role" . "user") ("content" . "Once upon a time, there was a")))
;;    (lambda (res _partialp) (message "@>>> %S" res))))

;; (defconst csv-4 (parse-csv-string-rows (f-read "~/Downloads/4 - Sheet1.csv") ?\, ?\" "\n"))

;; (nth 13 (car (cdr csv-4)))

;; (defconst llm (text-sage-llm-openai-chat-create :model "gpt-4"))

(provide 'text-sage)

;;; text-sage.el ends here

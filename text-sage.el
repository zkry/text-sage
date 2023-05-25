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

(defun text-sage-async-shell-command (callback args extract-response)
  "Run ARGS in a shell asynchronously and call CALLBACK with the output."
  (let* ((process (apply #'start-process
                         (append '("text-sage-llm-openai" "*text-sage-openai*") args))))

    (when process
      (with-current-buffer (process-buffer process)
        (erase-buffer))
      (set-process-sentinel
       process
       `(lambda (process _event)
          (let ((callback ,callback))
            (let ((output (with-current-buffer (process-buffer process)
                            (buffer-string)))
                  (exit-status (process-exit-status process)))
              (if (= exit-status 0)
                  (if (string-empty-p (string-trim output))
                      (funcall callback output nil)
                    (funcall callback (funcall ,extract-response output) nil))))))))))

(defun text-sage--filter-nil-plist-items (plist)
  "Return PLIST with nil items removed."
  (when plist
    (let ((key (car plist))
          (val (cadr plist)))
      (if val
          (cons key (cons val (text-sage--filter-nil-plist-items (cddr plist))))
        (text-sage--filter-nil-plist-items (cddr plist))))))


;;;; LLMs

(cl-defgeneric text-sage-llm-call (model prompt callback &optional stop)
  "Call Language MODEL with PROMPT and STOP and call CALLBACK with res.")

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
  (text-sage-async-shell-command
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
                      :logit_bias (text-sage-llm-openai-logit-bias model)
                      ))))
   (lambda (body)
     (gethash "text" (aref (gethash "choices" (json-parse-string body)) 0)))))

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
  "Call Language MODEL with PROMPT and STOP and call CALLBACK with res.")

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
  logit-bias)

(cl-defmethod text-sage-llm-chat-call ((model text-sage-llm-openai-chat) messages callback &optional stop)
  "Call OpenAI Language MODEL with MESSAGES and STOP and call CALLBACK with res."
  (text-sage-async-shell-command
   callback
   (list "curl"
         text-sage-llm-openai-chat-completions-url
         "--fail-with-body"
         "--no-progress-meter" "-m" "600"
         "-H" "Content-Type: application/json"
         "-H" (format "Authorization: Bearer %s" text-sage-openai-key)
         "-d" (json-encode
               (text-sage--filter-nil-plist-items
                (list :messages messages
                      :model (text-sage-llm-openai-chat-model model)
                      :max_tokens (text-sage-llm-openai-chat-max-tokens model)
                      :n (text-sage-llm-openai-chat-n model)
                      :temperature (text-sage-llm-openai-chat-temperature model)
                      :top_p (text-sage-llm-openai-chat-top-p model)
                      :stop stop
                      :presence_penalty (text-sage-llm-openai-chat-presence-penalty model)
                      :frequency_penalty (text-sage-llm-openai-chat-frequency-penalty model)
                      :logit_bias (text-sage-llm-openai-chat-logit-bias model)))))
   (lambda (body)
     (gethash "content" (gethash "message" (aref (gethash "choices" (json-parse-string body)) 0))))))

;;; Playground

(let ((llm (text-sage-llm-openai-create
            :model "text-davinci-003")))
  (text-sage-llm-call
   llm
   "Once upon a time, there was a"
   (lambda (res _partialp) (message ">>> %S" res))))

(let ((llm (text-sage-llm-hugging-face-create
            :model "stabilityai/stablelm-tuned-alpha-3b")))
  (text-sage-llm-call
   llm
   "You are an AI that only answers with one word. Question: True or false: the sky is blue. Response: "
   (lambda (res _partialp) (message "|>>> %S" res))))

(let ((llm (text-sage-llm-openai-chat-create
            :model "gpt-3.5-turbo")))
  (text-sage-llm-chat-call
   llm
   '((("role" . "user") ("content" . "Once upon a time, there was a")))
   (lambda (res _partialp) (message "@>>> %S" res))))



(provide 'text-sage)

;;; text-sage.el ends here

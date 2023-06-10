;;; text-sage-shell.el --- shell-maker integration for text-sage -*- lexical-binding: t -*-

(require 'esh-mode)
(require 'eshell)
(require 'ielm)
(require 'shell-maker)

(defun text-sage-shell-config-from-chain (chain)
  (make-shell-maker-config
   :name "text-sage"
   :validate-command
   (lambda (_command)
     nil)
   :execute-command
   (lambda (command _history callback error-callback)
     (text-sage-chain-run chain command callback))
   :on-command-finished
   (lambda (command output)
     ;; (chatgpt-shell--put-source-block-overlays)
     ;; (run-hook-with-args 'chatgpt-shell-after-command-functions
     ;;                     command output)
     )
   :redact-log-output
   (lambda (output)
     output)
   :prompt "text-sage> "
   :prompt-regexp (rx (seq bol "text-sage>" (or space "\n")))))

(defun text-sage-shell ()
  "Start a text-sage shell with a chain."
  (interactive)
  (let* ((llm (text-sage-llm-openai-chat-create :model "gpt-3.5-turbo" :max-tokens 3000))
         (memory (text-sage-conversation-buffer-memory-create))
         (chain (text-sage-conversation-chain-create
                 :prompt '((:system "you are chatbot")
                           :@history-messages
                           (:user "{{input}}"))
                 :memory memory
                 :llm llm)))
    (shell-maker-start (text-sage-shell-config-from-chain chain) nil)))

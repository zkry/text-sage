;;; text-sage-examples.el --- Example usages of text-sage

(require 'text-sage)

(let ((person-parser (text-sage-spec-parser '((name . string)
                                              (age . number)
                                              (country . (category "USA" "Iraq" "Iran" "Turkey" "Japan"))))))
  (text-sage-parse person-parser "{\"name\": \"Zachary Romero\",\n \"age\": 10,\n \"country\": \"Kazakhstan\"}"))

(let ((llm (text-sage-llm-openai-create :model "text-davinci-003" :max-tokens 100))
      (person-parser (text-sage-spec-parser '((name . string)
                                              (age . number)
                                              (country . (category "USA" "Iraq" "Iran" "Turkey" "Japan")))))
      (given "{\"name\": \"Zachary Romero\",\n \"age\": \"10\",\n \"country\": \"Iraqq\"}"))
  (condition-case err
      (text-sage-parse person-parser given)
    (error (text-sage-correct-parsing-error llm person-parser given (cadr err)
                                            (lambda (msg _)
                                              (message "CORRECTED: %s" msg))
                                            1))))


(let ((splitter (text-sage-character-text-splitter-create :chunk-size 27)))
  (text-sage-text-split splitter "This is an example text\n\nThis is another document!!!\n\nFunny :)"))

(let ((splitter (text-sage-recursive-character-splitter-create :chunk-size 60)))
  (text-sage-text-split
   splitter
   "This is a
funny document that has spacing
in weird places.

This is something that I don't know
how to handle
properly.

Another document is here. If it
goes too long it must split by the new lines.
I wonedr how it is going to handle this.
I have a long paragraph here.

and a word"))

(defconst testchain
  (let* ()
    chain))

(text-sage-conversation-chain-prompt testchain)

(text-sage-conversation-chain-memory testchain)
mydata
(text-sage-chain-run testchain "Output an example JSON." (lambda (msg _) (message ">>> %s" msg)))

(text-sage-format '((:system "YOu are a {{ adjective }} robot")
                         (:user "sunny {{ game }}"))
                       '((adjective . "funny")
                         (game . "baseball")))

(text-sage-chat-format '((:system "YOu are a {{ adjective }} robot")
                         :@history
                         (:user "sunny {{ game }}"))
                       '((adjective . "funny")
                         (game . "baseball")
                         (history . ((:user "tell me a joke")
                                     (:assistant "why did the chicken cross the road?")))))

(text-sage-chat-format

 '((input . "Give me an example data")))

(symbol-name :@elt)

(provide 'text-sage-examples)

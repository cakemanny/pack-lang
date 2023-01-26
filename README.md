# Pack Lang

Most definitely still a work in progress.

In this lisp I think I wanted to include a module system as this is
something that has been missing from languages I've implemented so far.

`.bin/pack` starts a REPL

Most of the ideas come from Clojure.


## Getting Started

In order to play with Pack in your python project.
Activate your virtual environment and then

```shell
pip install 'git+https://github.com/cakemanny/pack-lang'
```

```
$ pack
user=> (+ 1 2)
3
user=> 
```


## Example

```clojure
; create and initialise the "example" namespace
(ns example)
; import the python builtins module, saving it in the "py" namespace
; and referring to it in this oen
(import builtins)

; define a function taking a single parameter
; this opens a file, reads the contents, closes the file and then returns
; those contents
(defn read-file
  [file-name]
  (let [f ((. builtins open) file-name)
        contents ((. f read))]
    (do
      ((. f close))
      contents)))

;; we define a function that calls python's str split method
;; and then converts the list to a vector
(defn split- [sep the-str] (apply vector ((. the-str split) sep)))
;; we use that split- to make a utility function that can be curried
;; the argument after & is a rest param like "*args" in python
(defn split [sep & the-strs]
  (if the-strs
    (split- (first the-strs))
    (fn [a-str] (split- needle a-str))))

(def to-lines (split "\n"))
(defn to-words
  [lines]
  (map (split " ") lines))


(defmacro comment [& forms] nil)
(comment
  ; an example
  (example/read-file "/usr/share/calendar/calendar.lotr")
  (def s (example/read-file "/usr/share/calendar/calendar.lotr"))
  (to-lines s)
  ;; TODO: we need to implement map for vector
  (to-words (to-lines s))
)

;; vim:ft=clojure:
```

## Example with Flask

```shell
pip install Flask
```

```clojure
(ns example)
(import flask)
;; Flask requires a __name__ property, which is missing in interpreted mode
(do (ns pack.core) (def *compile* true) (ns example))

(def app
  ((. flask Flask) "example"))

(def health-endpoint
  (fn [] "Everything's gravy baby"))

;; We define a macro to comment out some pieces of code for the demo
(defmacro comment [& forms] nil)

;; Luckily this add_url_rule method accepts positional arguments
((. app add_url_rule)
 "/healthz"
 (comment endpoint goes here ... but let's leave it nil)
 health-endpoint)


;; code to paste into the repl during the demo
(comment
  (require 'example)
  ((. example/app run))
  )
; http://127.0.0.1:5000/healthz

;; vim:ft=clojure:
```


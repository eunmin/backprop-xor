(ns backprop-xor.core)

(def linearly-inseparable-samples [{:xs [0 0 1.0] :d 0} 
                                   {:xs [0 1.0 1.0] :d 1.0} 
                                   {:xs [1.0 0 1.0] :d 1.0}
                                   {:xs [1.0 1.0 1.0] :d 0}])

(def alpha 0.9)

(defn- sigmoid [x]
  (/ 1.0 (+ 1.0 (Math/exp (- x)))))

(defn- init-w []
  (dec (rand 2)))

(defn output [ws xs]
  (sigmoid (reduce + (map #(apply * %) (map vector ws xs)))))

(defn- output2 [ws xs]
  (let [ys (map #(output % xs) (:ws1 ws))]
    (output (:ws2 ws) ys)))

(defn backprop-xor [init-ws inputs]
  (reduce
    (fn [{:keys [ws1 ws2]} {:keys [xs d]}]
      (let [ys (map #(output % xs) ws1)
            y (output ws2 ys)
            e (- d y)
            delta (* y (- 1.0 y) e)
            e1 (map #(* delta %) ws2)
            delta1 (map #(* %2 (- 1.0 %2) %1) e1 ys)
            dw1s (map
                   #(map
                      (fn [delta1-delta]
                        (* alpha delta1-delta %))
                      xs)
                   delta1)
            dws (map #(* alpha delta %) ys)]
        {:ws1 (map (fn [ws dw1]
                     (map #(apply + %) (map vector ws dw1)))
                ws1
                dw1s)
         :ws2 (map #(apply + %) (map vector ws2 dws))}))
    init-ws
    inputs))

(defn train [n samples]
  (backprop-xor
    {:ws1 [[(init-w) (init-w) (init-w)]
           [(init-w) (init-w) (init-w)]
           [(init-w) (init-w) (init-w)]
           [(init-w) (init-w) (init-w)]]
     :ws2 [(init-w) (init-w) (init-w) (init-w)]}
    (flatten (repeat n samples))))

(defn -main [& args]
  (let [ws (train 1000 linearly-inseparable-samples)
        xss (map :xs linearly-inseparable-samples)]
    (println "target ouput     : " (map :d linearly-inseparable-samples))
    (println "inference output : " (map #(output2 ws %) xss))))

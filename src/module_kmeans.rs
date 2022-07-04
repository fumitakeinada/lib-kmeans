
pub mod kmeans {
    use ndarray::prelude::*;
    use ndarray::{Array,ArrayView1,Array1,Array2,ShapeError,Axis};
    use itertools::Itertools;


    use rand::{thread_rng, Rng};
    use rand::distributions::{Uniform};
    use rand_distr::{Distribution};
    use std::thread;
    use std::sync::{Arc, Mutex};
    use rayon::prelude::*;
    use serde::{Serialize, Deserialize};

    pub struct KMeans {
        n_clusters: usize, // clusters number
        max_iter: usize, 
        cols: usize, // 学習データの次元数（列数）
        mean_points: Array2<f64>,
    }

    impl KMeans {
        pub fn new(n_clusters:usize, max_iter:usize) -> Self{
            KMeans {
                n_clusters: n_clusters, 
                max_iter: max_iter,
                cols: 0,
                mean_points: Array::zeros((0, 0)),
            }
        }

        pub fn fit(& mut self, x:&Array2<f64>) -> Vec<usize> {
            self.cols = x.ncols();

            // データに対して、クラスタをランダムに割り当て
            let mut labels:Vec<usize> = 
                (0..x.nrows()).map(|_| {
                    rand::thread_rng().gen_range(0..self.n_clusters)
                }).collect();
  
            // 前回のクラスタ割り当て保存領域（データ数を0で初期化）
            let mut pre_labels = vec![0; x.nrows()];

            // 各データに属しているクラスタが変化しなくなるか、一定回数を繰り返して終了
            for _ in 0..self.max_iter {
                 // 一つ前のクラスタラベルの割り当てと比較
                 let mut count = 0;
                 for i in 0..labels.len(){
                     if labels[i] == pre_labels[i] {
                         count = count + 1;
                     }
                 }
                 // 全てのラベルが前回と同じ場合は終了
                 if count == x.nrows() {
                     break;
                 }

                 // 次の計算のためにクラスタのラベルを保存
                 pre_labels = labels.clone();
                 
                // クラスタの重心計算
                // 重心のArray2を作成（行はn_clusters数だけ）
                let mut mean_array: Array2<f64> = Array::zeros((0,x.ncols()));
                
                for i in 0..self.n_clusters {
                    // テンポラリのArray2を作成
                    let mut tmp_array: Array2<f64> = Array::zeros((0,x.ncols())); 
            
                    // クラスタに属するindexのイテレータを用意
                    let cluster_set_iter = labels.iter().positions(|v| v==&i);
                    for j in cluster_set_iter{
                        // テンポラリに追加
                        tmp_array.push_row(x.slice(s![j, 0..x.ncols()]));
                    }

                    // テンポラリ内の平均を算出し重心とし、重心のArray2に加える
                    let tmp = tmp_array.mean_axis(Axis(0));
                    match tmp {
                        None => {},
                        Some(ret) => {mean_array.push_row(ret.view());},
                    }
                    
                }
                // labelsの再初期化
                labels = self.get_labels(x, &mean_array);

                // 各重心の保存
                self.mean_points = mean_array;
            }

            labels       
        }
        
        // 推論（どの重心に近いかの区分）
        pub fn predict(& mut self, x:&Array2<f64>) -> Vec<usize> {
            let mean_points = self.mean_points.clone();
            self.get_labels(x, &mean_points)
        }

        fn get_dist_mean(& mut self, x:&Array2<f64>, mean_points:&Array2<f64>) -> Array2<f64> {
            let mut dist_array:Array2<f64> = Array::zeros((0, x.nrows()));
            for i in 0..mean_points.nrows(){
                let dist_array_tmp:Array2<f64> = x - &mean_points.slice(s![i, 0..mean_points.ncols()]);
                let dist = (&dist_array_tmp*&dist_array_tmp).sum_axis(Axis(1)).mapv_into(|v| v.sqrt());
                dist_array.push_row(dist.view());
            }
            dist_array
        }

        fn get_labels(& mut self, x:&Array2<f64>, mean_points:&Array2<f64>) -> Vec<usize> {
            // データ毎の各重心との距離を算出
            let dist_array:Array2<f64> = self.get_dist_mean(x, mean_points);
                
            // 最小クラスの割り当て
            let labels:Vec<usize> = (0..dist_array.ncols()).map(|i| {
                let v = dist_array.slice(s![0..dist_array.nrows(), i]).to_vec();
        
                let (min_index, min) = v.iter()
                    .enumerate()
                    .fold((usize::MIN, f64::MAX), |(i_a, a), (i_b, &b)| {
                        if b < a {
                            (i_b, b)
                        } else {
                            (i_a, a)
                        }
                });
                min_index               
            }).collect();
            labels
        }
    }

}
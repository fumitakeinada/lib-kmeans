
pub mod kmeans {
    use ndarray::prelude::*;
    use ndarray::{Array,Array2,Axis,ShapeError};
    use itertools::Itertools;

    use rand::Rng;

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

        pub fn fit(& mut self, x:&Array2<f64>) -> Result<Vec<usize>, ShapeError> {
            self.cols = x.ncols();

            // データに対して、クラスタをランダムに割り当て
            let mut labels:Vec<usize> = self.init_labels(&x, self.n_clusters);

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

                    // 対象のindexに属するデータのみをテンポラリに追加
                    for j in cluster_set_iter{
                        match tmp_array.push_row(x.slice(s![j, ..])) {
                            Ok(_) => {},
                            Err(e) => {return Err(e)},
                        }
                    }

                    // テンポラリ内の平均を算出し重心とし、重心のArray2に加える
                    match tmp_array.mean_axis(Axis(0)) {
                        None => {},
                        Some(ret) => {
                            match mean_array.push_row(ret.view()) {
                                Ok(_) => {},
                                Err(e) => {return Err(e)},
                            }
                        },
                    }
                    
                }
                // labelsの再割り当て
                labels = match self.get_labels(x, &mean_array){
                    Ok(r) => {r},
                    Err(e) => {return Err(e)},
                };

                // 各重心の保存
                self.mean_points = mean_array;
            }

            Ok(labels)
        }
        
        // 推論（どの重心に近いかの区分）
        pub fn predict(& mut self, x:&Array2<f64>) -> Result<Vec<usize>, PredictError> {
            // 次元数の検査
            if self.cols != x.ncols() {
                return Err(PredictError::DimNumError(x.ncols()))
            }


            let mean_points = self.mean_points.clone();
            match self.get_labels(x, &mean_points) {
                Ok(r) => {return Ok(r)},
                Err(e) => {return Err(PredictError::ShapeError)},
            };
        }

        // Get dimension number.
        pub fn get_dim(&mut self) -> usize {
            self.cols
        }


        fn init_labels(& mut self, x:&Array2<f64>, n_clusters:usize) -> Vec<usize> {
            (0..x.nrows()).map(|_| {
                rand::thread_rng().gen_range(0..n_clusters)
            }).collect()
        }

        fn get_dist_mean(& mut self, x:&Array2<f64>, mean_points:&Array2<f64>) -> Result<Array2<f64>, ShapeError> {
            let mut dist_array:Array2<f64> = Array::zeros((0, x.nrows()));
            for i in 0..mean_points.nrows(){
                let dist_array_tmp:Array2<f64> = x - &mean_points.slice(s![i, ..]);
                let dist = (&dist_array_tmp*&dist_array_tmp).sum_axis(Axis(1)).mapv_into(|v| v.sqrt());
                match dist_array.push_row(dist.view()) {
                    Ok(_) => {},
                    Err(e) => return Err(e)
                };
            }
            Ok(dist_array)
        }

        fn get_labels(& mut self, x:&Array2<f64>, mean_points:&Array2<f64>) -> Result<Vec<usize>, ShapeError> {
            // データ毎の各重心との距離を算出
            let dist_array:Array2<f64> = match self.get_dist_mean(x, mean_points){
                Ok(r) => {r},
                Err(e) => return Err(e)
            };

                
            // 最小クラスの割り当て
            let labels:Vec<usize> = (0..dist_array.ncols()).map(|i| {
                let v = dist_array.slice(s![.., i]).to_vec();
                let (min_index, _min) = v.iter()
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
            Ok(labels)
        }
    }

    // 推論用のエラー定義
    #[derive(Debug)]
    pub enum PredictError {
        ShapeError,
        DimNumError(usize), // 次元数エラー
    }

}


pub mod kmeans {
    use ndarray::prelude::*;
    use ndarray::{Array,Array2,Axis,ShapeError};
    use itertools::Itertools;

    use rand::Rng;

    use serde::{Serialize, Deserialize};
    
    pub trait KMeansModel {
        // 初期化
        fn new(n_clusters: usize, max_iter:usize) -> Self;

        // 学習データに対する学習
        fn fit(&mut self, x:&Array2<f64>) -> Result<Vec<usize>, ShapeError>;
        
        // 推論データに対する予測（どの重心に近いかの区分）
        fn predict(&mut self, x:&Array2<f64>) -> Result<Vec<usize>, PredictError> {
            // 次元数の検査
            if self.get_dim() != x.ncols() {
                return Err(PredictError::DimNumError(x.ncols()))
            }

            //let mean_points = self.mean_points.clone();
            let mean_points = self.get_mean_points();
            match get_labels(x, &mean_points) {
                Ok(r) => {return Ok(r)},
                Err(e) => {return Err(PredictError::ShapeError)},
            };
        }

        // 学習データに対する次元
        fn get_dim(& self) -> usize;

        // 学習データに対する重心の取得
        fn get_mean_points(&self) -> Array2<f64>;

    }

    pub struct KMeans {
        n_clusters: usize, // clusters number
        max_iter: usize, 
        dim: usize, // 学習データの次元数（列数）
        mean_points: Array2<f64>,
    }
    
    impl KMeansModel for KMeans {
        fn new(n_clusters:usize, max_iter:usize) -> Self{
            Self {
                n_clusters: n_clusters, 
                max_iter: max_iter,
                dim: 0,
                mean_points: Array::zeros((0, 0)),
            }
        }

        fn fit(& mut self, x:&Array2<f64>) -> Result<Vec<usize>, ShapeError> {
            let dim:usize = x.ncols();
            let n_clusters:usize = self.n_clusters;
            let max_iter = self.max_iter;
            self.dim = dim;

            // データに対して、クラスタをランダムに割り当て
            let init_labels:Vec<usize> = init_labels(&x, n_clusters);

            // 重心を計算し、重心とクラスタを得る。
            match fit_clusters(x, init_labels, n_clusters, max_iter) {
                Ok(r) => {
                    self.mean_points = r.1;
                    Ok(r.0)
                },
                Err(e) => {return Err(e)},
            }
        }
        
        // Get dimension number.
        fn get_dim(&self) -> usize {
            self.dim
        }

        // 学習された重心の取得
        fn get_mean_points(&self) -> Array2<f64> {
            self.mean_points.clone()
        }

    }

    // クラスタの割り当て（初期値はあらかじめ用意）
    fn fit_clusters(x:&Array2<f64>, init_labels:Vec<usize>, n_clusters:usize, max_iter:usize ) -> Result<(Vec<usize>, Array2<f64>), ShapeError>{

        let data_num:usize = x.nrows();
        let data_dim:usize = x.ncols();

        let mut labels : Vec<usize> = init_labels.clone();

        let mut pre_labels = vec![0; data_num];

        let mut mean_array: Array2<f64> = Array::zeros((0, data_dim));

        // 各データに属しているクラスタが変化しなくなるか、一定回数を繰り返して終了
        for _ in 0..max_iter {
             // 一つ前のクラスタラベルの割り当てと比較
             let mut count = 0;
             for i in 0..labels.len(){
                 if labels[i] == pre_labels[i] {
                     count = count + 1;
                 }
             }
             // 全てのラベルが前回と同じ場合は終了
             if count == data_num {
                 break;
             }

             // 次の計算のためにクラスタのラベルを保存
            pre_labels = labels.clone();
             
            // クラスタの重心計算
            // 重心のArray2を作成（行はn_clusters数だけ）
            mean_array = Array::zeros((0, data_dim));
            
            for i in 0..n_clusters {
                // テンポラリのArray2を作成
                let mut tmp_array: Array2<f64> = Array::zeros((0, data_dim)); 
        
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
            labels = match get_labels(x, &mean_array){
                Ok(r) => {r},
                Err(e) => {return Err(e)},
            };

        }

        Ok((labels, mean_array))

    }

    // クラスタの初期値を割り当て(KMeans)
    fn init_labels(x:&Array2<f64>, n_clusters:usize) -> Vec<usize> {
        (0..x.nrows()).map(|_| {
            rand::thread_rng().gen_range(0..n_clusters)
        }).collect()
    }

    // 重心との距離の取得
    fn get_dist_mean(x:&Array2<f64>, mean_points:&Array2<f64>) -> Result<Array2<f64>, ShapeError> {
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

    // クラスタの割り当てを取得
    fn get_labels(x:&Array2<f64>, mean_points:&Array2<f64>) -> Result<Vec<usize>, ShapeError> {
        // データ毎の各重心との距離を算出
        let dist_array:Array2<f64> = match get_dist_mean(x, mean_points){
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

    // 推論用のエラー定義
    #[derive(Debug)]
    pub enum PredictError {
        ShapeError,
        DimNumError(usize), // 次元数エラー
    }

}

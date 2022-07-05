pub mod module_kmeans;
#[macro_use(s)]
extern crate ndarray;
extern crate rand;
extern crate rayon;
extern crate statrs;


#[cfg(test)]
mod tests {
    use ndarray::{Array,Array2,Axis, concatenate};
    use super::*;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::{Normal, Uniform};

        // Make training data
    fn make_train_data(data_num:usize, mean:f64, std_dev:f64, dim: usize) -> Array2<f64>{
        let clusters_num:usize = 4;
        // Multiple normal distributions
        let train1:Array2<f64> = Array::random((data_num/clusters_num , dim), Normal::new(mean * 0.3 + 2.0, std_dev * 0.3).unwrap());
        let train2:Array2<f64> = Array::random((data_num/clusters_num, dim), Normal::new(mean * 0.5 - 2.0, std_dev * 0.5).unwrap());
        let train3:Array2<f64> = Array::random((data_num/clusters_num , dim), Normal::new(mean * 0.1 + 12.0, std_dev * 0.1).unwrap());
        let train4:Array2<f64> = Array::random((data_num/clusters_num, dim), Normal::new(mean * 0.2 - 20.0, std_dev * 0.3).unwrap());
        let arr_train = concatenate(Axis(0), &[train1.view(), train2.view(), train3.view(), train4.view()]).unwrap();
        
        arr_train
    }

    #[test]
    fn fit_test() {
        // Make normal training data
        let mean:f64 = 0.0; // mean
        let std_dev:f64 = 1.0; // standard deviation
        let dim = 2; // dimension
        let data_num = 200;

        let train_data = make_train_data(data_num, mean, std_dev, dim );
        
        let mut km = module_kmeans::kmeans::KMeans::new(4, 50);
        let labels = km.fit(&train_data).unwrap();
        println!("train_data:{:?}", labels);
        assert_eq!(labels.len(), data_num);
    }

    #[test]
    fn predict_test() {
        // Make normal training data
        let mean:f64 = 0.0; // mean
        let std_dev:f64 = 1.0; // standard deviation
        let dim = 2; // dimension
        let data_num = 2000;

        let train_data = make_train_data(data_num, mean, std_dev, dim );
        let mut km = module_kmeans::kmeans::KMeans::new(4, 50);
        let train_labels = km.fit(&train_data).unwrap();
        println!("train_data for predict:{:?}", train_labels);
        assert_eq!(train_labels.len(), data_num);

        let test_labels = match km.predict(&train_data){
            Ok(r) => r,
            Err(e) => {panic!("Error: {:?}",e);}
        };

        println!("test_data for predict:{:?}", test_labels);
        assert_eq!(test_labels.len(), data_num);

        assert_eq!(train_labels, test_labels);

    }
}

/*Deformation Transfer using TriMesh2
 * */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <utility>
#include <set> 
#include <map>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <Eigen/CholmodSupport>

#include <flann/flann.hpp>
#include <boost/graph/graph_concepts.hpp>

#include "TriMesh.h"
#include "TriMesh_algo.h"

using namespace std;
using namespace trimesh;

#define THRESHOLD_DIST 0.1     //less
#define THRESHOLD_NORM 0.7     //greater

static double* dataset;
static flann::Index<flann::L2<double>> *kd_flann_index;


//build kd tree for the dst
static void buildKdTree(trimesh::TriMesh* mesh)
{
	std::vector<trimesh::point> verts = mesh->vertices;
	//std::vector<trimesh::TriMesh::Face> faces = mesh->faces;
	
	int vertNum=verts.size();
	dataset= new double[3*vertNum];
	
	for(int i=0; i<vertNum; i++)
	{
		dataset[3*i]=verts[i][0];
		dataset[3*i+1]=verts[i][1];
		dataset[3*i+2]=verts[i][2];
	}
	
	flann::Matrix<double> flann_dataset(dataset,vertNum,3);
	kd_flann_index=new flann::Index<flann::L2<double> >(flann_dataset,flann::KDTreeIndexParams(1));
	kd_flann_index->buildIndex();
	return;
}

static void releaseKdTree()
{
	delete[] dataset;
	delete kd_flann_index;
	kd_flann_index=NULL;
}

static void getKdCorrespondences(trimesh::TriMesh* src, trimesh::TriMesh* dst,
							std::vector<std::pair<int,int> >& soft_corres)
{
	//cout<<"###"<<endl;
	
	const int knn=1;
	flann::Matrix<double> query(new double[3],1,3);
	flann::Matrix<int>    indices(new int[query.rows*knn],query.rows,knn);
	flann::Matrix<double> dists(new double[query.rows*knn],query.rows,knn);
	
	soft_corres.clear();
	for(int i=0; i<src->vertices.size();i++)
	{
		
		
		for(int j=0; j<3;j++)
			query[0][j]=src->vertices[i][j];
		
		kd_flann_index->knnSearch(query,indices,dists,1,flann::SearchParams(flann::FLANN_CHECKS_UNLIMITED));
		
		
		Eigen::Vector3d normal1(src->normals[i][0],src->normals[i][1],src->normals[i][2]);
		Eigen::Vector3d normal2(dst->normals[indices[0][0]][0],dst->normals[indices[0][0]][1],dst->normals[indices[0][0]][2]);
		
		//cout<<i<<" "<<indices[0][0]<<endl;
		
		if(dists[0][0]<THRESHOLD_DIST && normal1.dot(normal2)>=THRESHOLD_NORM)
		{
			soft_corres.push_back(std::make_pair(i,indices[0][0]));
			//cout<<i<<" "<<indices[0][0]<<endl;
		}
	}
	
	delete[] query.ptr();
	delete[] indices.ptr();
	delete[] dists.ptr();
	
	return;
}


void solveLinearSystem(trimesh::TriMesh* src, trimesh::TriMesh* dst,
					const std::vector<std::pair<int,int> >& corres,
					double weights[4])
{
	int vertNum=src->vertices.size();
	int faceNum=src->faces.size();
	
	int rows =0;
	int cols = vertNum + faceNum;
	
//	cout<<faceNum<<endl;
	
	std::vector<std::pair<int,int> > soft_corres;
	getKdCorrespondences(src, dst,soft_corres);
	
	//cout<<soft_corres.size()<<endl;
	
	for(int i=0; i<faceNum; i++)
	{
		//cout<<i<<" ";
		for(int j=0; j<3;j++)
		{
			if(src->across_edge[i][j]!=-1)
				rows+=3;
		}
	}
		
	
	rows+=faceNum*3;
	rows+=corres.size();
	rows+=soft_corres.size();
	
	//cout<<rows<<endl;
	
	
	Eigen::SparseMatrix<double> A(rows,cols);
	A.reserve(Eigen::VectorXi::Constant(cols,200));
	std::vector<Eigen::VectorXd> Y(3,Eigen::VectorXd(rows));
	Y[0].setZero();
	Y[1].setZero();
	Y[2].setZero();
	
	//cout<<faceNum<<endl;
	std::vector<Eigen::Matrix3d> Q_hat_inverse(faceNum);
	Eigen::Matrix3d _inverse;
	for(int i=0; i<faceNum;i++)
	{
		trimesh::TriMesh::Face index_i = src->faces[i];
		trimesh::vec v[3];
		v[0]= src->trinorm(i);
		//if(i==0)cout<<v[0]<<endl;
		v[1]=src->vertices[index_i[1]]-src->vertices[index_i[0]]; //v2-v1
		v[2]=src->vertices[index_i[2]]-src->vertices[index_i[0]]; //v3-v1
		
		for(int k=0; k<3;k++)
		{
			for(int j=0; j<3; j++)
			{
				Q_hat_inverse[i](k,j)=v[j][k];
				
			}
		}
		
		//if(i==0)cout<<Q_hat_inverse[i]<<endl;
		
		_inverse=Q_hat_inverse[i].inverse();
		Q_hat_inverse[i]=_inverse;
	}
	
	
	
	int energy_size[4]={0,0,0,0};
	
	//linear system
	double weight_smooth=weights[0];
	int row=0;
	for(int i=0; i<faceNum; i++)
	{
		//cout<<i<<endl;
		Eigen::Matrix3d Q_i_hat=Q_hat_inverse[i];
		trimesh::TriMesh::Face index_i = src->faces[i];
		
		//cout<<"across"<<src->across_edge[i]<<endl;
		for(size_t _j=0; _j<src->across_edge[i].size(); _j++)
		{
			int j=src->across_edge[i][_j];
			if(j==-1)continue;
			//if(i<10)cout<<j<<endl;
			Eigen::Matrix3d Q_j_hat=Q_hat_inverse[j];
			trimesh::TriMesh::Face index_j = src->faces[j];
			
			for(int k=0; k<3; k++)
			{
				for(int p=0; p<3;p++)
				{
					A.coeffRef(row+k,faceNum + index_i[p])=0.0;
				}
			}
			
				
			for(int k=0; k<3; k++)
			{
				A.coeffRef(row+k, i) = weight_smooth*Q_i_hat(0,k);
				A.coeffRef(row+k, faceNum + index_i[0]) += -weight_smooth*(Q_i_hat(1,k)+Q_i_hat(2,k));
				A.coeffRef(row+k, faceNum + index_i[1]) +=  weight_smooth* Q_i_hat(1, k);
                A.coeffRef(row+k, faceNum + index_i[2]) +=  weight_smooth* Q_i_hat(2, k);

                A.coeffRef(row+k, j) = -weight_smooth*Q_j_hat(0,k);   //n
                A.coeffRef(row+k, faceNum + index_j[0]) +=  weight_smooth*(Q_j_hat(1, k)+Q_j_hat(2, k));
                A.coeffRef(row+k, faceNum + index_j[1]) += -weight_smooth* Q_j_hat(1, k);
                A.coeffRef(row+k, faceNum + index_j[2]) += -weight_smooth* Q_j_hat(2, k);
			}
			row+=3;
		}
	}
	energy_size[0] = row;
	

	
	double weight_regular = weights[1];
    for (int i=0; i<corres.size(); ++i, ++row) 
	{
		A.coeffRef(row, faceNum + corres[i].first) = weight_regular;
        for (int j=0; j<3; ++j)
            Y[j](row) = weight_regular*dst->vertices[corres[i].second][j];
    }
	energy_size[1] = row;

	
	
	
	double weight_identity = weights[2];
    for (int i=0; i<faceNum; ++i) 
	{
        Eigen::Matrix3d Q_i_hat = Q_hat_inverse[i];
		trimesh::TriMesh::Face index_i = src->faces[i];
		
        Y[0](row) = weight_identity;    Y[0](row+1) = 0.0;              Y[0](row+2) = 0.0; 
        Y[1](row) = 0.0;                Y[1](row+1) = weight_identity;  Y[1](row+2) = 0.0; 
        Y[2](row) = 0.0;                Y[2](row+1) = 0.0;              Y[2](row+2) = weight_identity; 
		
        for (int k=0; k<3; ++k, ++row) 
		{
            A.coeffRef(row, i) = weight_identity*Q_i_hat(0, k);   //n
            A.coeffRef(row, faceNum + index_i[0]) = -weight_identity*(Q_i_hat(1, k) + Q_i_hat(2, k));
            A.coeffRef(row, faceNum + index_i[1]) = weight_identity*Q_i_hat(1, k);
            A.coeffRef(row, faceNum + index_i[2]) = weight_identity*Q_i_hat(2, k);
        }
    }
	energy_size[2] = row;
	
	

    double weight_soft_constraint = weights[3];
    for (int i=0; i<soft_corres.size(); ++i, ++row) 
	{
		A.coeffRef(row, faceNum + soft_corres[i].first) = weight_soft_constraint;
        for (int j=0; j<3; ++j)
            Y[j](row) = weight_soft_constraint*dst->vertices[soft_corres[i].second][j];
    }
	energy_size[3] = row;
	

	//cout<<row<<endl;

   	//start solving the least-square problem
	fprintf(stdout, "finished filling matrix\n");
	Eigen::SparseMatrix<double> At = A.transpose();
	Eigen::SparseMatrix<double> AtA = At*A;

//  	Eigen::SimplicialCholesky<Eigen::SparseMatrix<double> > solver;
//  	solver.compute(AtA);
	
	Eigen::CholmodSupernodalLLT< Eigen::SparseMatrix<double> > solver;
	solver.compute(AtA);
	
	
	if (solver.info() != Eigen::Success) {
		fprintf(stdout, "unable to defactorize AtA\n");
		exit(-1);
	}

    Eigen::VectorXd X[3];
    for (int i=0; i<3; ++i)
	{
        Eigen::VectorXd AtY = At*Y[i];
        X[i] = solver.solve(AtY);
		Eigen::VectorXd Energy = A*X[i] - Y[i];
		Eigen::VectorXd smoothEnergy      = Energy.head(energy_size[0]);
		Eigen::VectorXd hardRegularEnergy = Energy.segment(energy_size[0], energy_size[1]-energy_size[0]);
		Eigen::VectorXd identityEnergy    = Energy.segment(energy_size[1], energy_size[2]-energy_size[1]);
		Eigen::VectorXd softRegularEnergy = Energy.tail(energy_size[3]-energy_size[2]);
		fprintf(stdout, "\t%lf = %lf + %lf + %lf + %lf\n", 
			Energy.dot(Energy), smoothEnergy.dot(smoothEnergy), hardRegularEnergy.dot(hardRegularEnergy), 
			identityEnergy.dot(identityEnergy), softRegularEnergy.dot(softRegularEnergy));
    }
    
    //fill data back to src
    for (int i=0; i<faceNum; ++i)
        for (int d=0; d<3; ++d)
            //src->normals[i][d] = X[d](i);
		
    for (int i=0; i<vertNum; ++i) 
        for (int d=0; d<3; ++d)
            src->vertices[i][d] = X[d](faceNum+i);
    
    return;
}


void deformTransfer(trimesh::TriMesh* src, trimesh::TriMesh* dst,
					const std::vector<std::pair<int,int> >& corres)
{
	
	buildKdTree(dst);
	//smooth, regular, identity weights, soft_constraints(nearest point)
    double weights[4] = {10.0, 100.0, 0.1, 0.0};    //{1.0, 10000, 0.1, 0.0};  
	
	int cnt = 0;    //change the step vale and upper bound of w[3] according to the data.
	while(weights[3]<=200.0)
	{
		solveLinearSystem(src, dst, corres, weights);
		string filename="src_to_dst_"+to_string(cnt++)+".obj";
		src->write(filename);
		
		weights[3]+=20.0;
	}

	releaseKdTree();
	return;
}

int main()
{
	string srcFilename="../data/ref_horse.obj";
	string dstFilename="../data/ref_camel.obj";
	
	trimesh::TriMesh *src, *dst;
	src=trimesh::TriMesh::read(srcFilename);
	dst=trimesh::TriMesh::read(dstFilename);
	
	src->need_neighbors();
	src->need_normals();
	src->need_across_edge();
	
	dst->need_neighbors();
	dst->need_normals();
	dst->need_across_edge();
	
	std::vector<std::pair<int,int> > corres;
	string corresFilename="../data/ref_horse_camel.cons";
	ifstream fin(corresFilename);
	int cnt;
	fin>>cnt;
	corres.resize(cnt);
	for(int i=0; i<cnt; i++)
	{
		fin>>corres[i].first>>corres[i].second;
	}
	
	
// 	double dist=0.0;
// 	for(size_t i=0; i<corres.size(); i++)
// 	{
// 		trimesh::point src_pt=src->vertices[corres[i].first];
// 		trimesh::point dst_pt=dst->vertices[corres[i].second];
// 		
// 		Eigen::Vector3d v=Eigen::Vector3d(src_pt[0],src_pt[1],src_pt[2])- \
// 						  Eigen::Vector3d(dst_pt[0],dst_pt[1],dst_pt[2]);
// 						  
// 		dist+=v.norm();
// 	}
// 	cout<<dist/corres.size()<<endl;


	deformTransfer(src,dst,corres);
	
	return 0;
}
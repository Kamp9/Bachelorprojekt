# coding=utf-8
import numpy as np
import substitution


def lu_inplace(A):
    (m, n) = A.shape
    A = A.astype(np.float64)
    for k in range(m-1):
        A[k+1:, k] = (1.0 / A[k, k]) * A[k+1:, k]  # division med 0 er muligt
        A[k+1:, k+1:] = A[k+1:, k+1:] - A[k+1:, k, np.newaxis] * A[k, k+1:]
    L = np.tril(A)
    np.fill_diagonal(L, 1)
    return L, np.triu(A)


def lu_out_of_place(A):
    (m, n) = A.shape
    A = A.astype(np.float64)
    U = np.zeros((m, m))
    L = np.identity(m)
    for k in range(m):
        U[k, k:] = A[k, k:]
        L[k+1:, k] = (1.0 / A[k, k]) * A[k+1:, k]  # division med 0 er muligt
        A[k+1:, k+1:] = A[k+1:, k+1:] - L[k+1:, k, np.newaxis] * U[k, k+1:]
    return L, U


def solve(A, b):
    L, U = lu_inplace(A)  # kan også være lu_out_of_place
    z = substitution.forward_substitution(L, b)
    x = substitution.backward_substitution(U, z)
    return x


def lu_partial_pivot(A):
    (m, n) = A.shape
    L = np.identity(m)
    P = np.identity(m)  # kan være P = L
    U = A.astype(np.float64)
    for k in range(m):
        pivot = k + np.absolute(U[k:, k]).argmax(axis=0)
        if k != pivot:
            temp = U[k, :].copy()  # kan vi undgå copy()?
            U[k, :] = U[pivot, :]
            U[pivot, :] = temp
            temp = P[k, :].copy()  # kan vi undgå copy()?
            P[k, :] = P[pivot, :]
            P[pivot, :] = temp
            if k >= 1:
                temp = L[k, :k].copy()  # kan vi undgå copy()?
                L[k, :k] = L[pivot, :k]
                L[pivot, :k] = temp
        L[k+1:, k] = (1.0 / U[k, k]) * U[k+1:, k]
        U[k+1:, k+1:] = U[k+1:, k+1:] - L[k+1:, k, np.newaxis] * U[k, k+1:]
    return P, L, np.triu(U)



"""
[n,n]=size(A);
L=eye(n); P=L; U=A;
for k=1:n
    [pivot m]=max(abs(U(k:n,k)));
    m=m+k-1;
    if m~=k
        % interchange rows m and k in U
        temp=U(k,:);
        U(k,:)=U(m,:);
        U(m,:)=temp;
        % interchange rows m and k in P
        temp=P(k,:);
        P(k,:)=P(m,:);
        P(m,:)=temp;
        if k >= 2
            temp=L(k,1:k-1);
            L(k,1:k-1)=L(m,1:k-1);
            L(m,1:k-1)=temp;
        end
    end
    for j=k+1:n
        L(j,k)=U(j,k)/U(k,k);
        U(j,:)=U(j,:)-L(j,k)*U(k,:);
    end
end
"""

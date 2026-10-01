import axios, { AxiosError } from 'axios';

const instance = axios.create({
  baseURL: import.meta.env.REACT_APP_RUUTER_PRIVATE_API_URL + '/generic/',
  headers: {
    Accept: 'application/json',
    'Content-Type': 'application/json',
  },
  withCredentials: true,
});

instance.interceptors.response.use(
  (axiosResponse) => {
    return axiosResponse;
  },
  (error: AxiosError) => {
    return Promise.reject(new Error(error.message));
  }
);

export default instance;

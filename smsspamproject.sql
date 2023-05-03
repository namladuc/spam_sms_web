-- phpMyAdmin SQL Dump
-- version 5.2.0
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: May 03, 2023 at 03:30 PM
-- Server version: 10.4.21-MariaDB
-- PHP Version: 8.0.10

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `smsspamproject`
--

-- --------------------------------------------------------

--
-- Table structure for table `class`
--

CREATE TABLE `class` (
  `class_id` int(11) NOT NULL,
  `label` varchar(10) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `class`
--

INSERT INTO `class` (`class_id`, `label`) VALUES
(0, 'Ham'),
(1, 'Spam');

-- --------------------------------------------------------

--
-- Table structure for table `data_group_info`
--

CREATE TABLE `data_group_info` (
  `id_dgroup` int(11) NOT NULL,
  `group_name` varchar(100) DEFAULT NULL,
  `tfidf_path` varchar(200) NOT NULL,
  `test_size` float NOT NULL DEFAULT 0.2
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `data_group_info`
--

INSERT INTO `data_group_info` (`id_dgroup`, `group_name`, `tfidf_path`, `test_size`) VALUES
(1, 'SMS Spam Collections Original', 'static/tfidf//tfidf_1.pickle', 0.3);

-- --------------------------------------------------------

--
-- Table structure for table `data_group_split`
--

CREATE TABLE `data_group_split` (
  `id_dtrain` int(11) NOT NULL,
  `id_dgroup` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- --------------------------------------------------------

--
-- Table structure for table `data_input`
--

CREATE TABLE `data_input` (
  `id` int(11) NOT NULL,
  `id_user` int(11) DEFAULT NULL,
  `original_text` varchar(255) DEFAULT NULL,
  `create_at` datetime DEFAULT NULL,
  `create_by` varchar(100) DEFAULT NULL,
  `update_at` datetime DEFAULT NULL,
  `update_by` varchar(100) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- --------------------------------------------------------

--
-- Table structure for table `data_train`
--

CREATE TABLE `data_train` (
  `id_dtrain` int(11) NOT NULL,
  `text` varchar(255) DEFAULT NULL,
  `class_id` int(11) DEFAULT NULL,
  `create_at` datetime DEFAULT NULL,
  `create_by` varchar(100) DEFAULT NULL,
  `update_at` datetime DEFAULT NULL,
  `update_by` varchar(100) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- --------------------------------------------------------

--
-- Table structure for table `model_info`
--

CREATE TABLE `model_info` (
  `id_model` int(11) NOT NULL,
  `model_name` varchar(200) DEFAULT NULL,
  `model_class` varchar(200) NOT NULL,
  `description` varchar(255) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `model_info`
--

INSERT INTO `model_info` (`id_model`, `model_name`, `model_class`, `description`) VALUES
(1, 'Support Vector Classification', 'Sklearn', 'Default SVC Class');

-- --------------------------------------------------------

--
-- Table structure for table `model_train`
--

CREATE TABLE `model_train` (
  `id_train` int(11) NOT NULL,
  `id_model` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- --------------------------------------------------------

--
-- Table structure for table `model_train_state`
--

CREATE TABLE `model_train_state` (
  `id_train` int(11) NOT NULL,
  `id_dgroup` int(11) DEFAULT NULL,
  `path_to_state` varchar(255) DEFAULT NULL,
  `can_use` tinyint(1) DEFAULT NULL,
  `time_train` float DEFAULT NULL,
  `create_at` datetime DEFAULT current_timestamp(),
  `create_by` varchar(100) DEFAULT NULL,
  `update_at` datetime DEFAULT current_timestamp(),
  `update_by` varchar(100) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- --------------------------------------------------------

--
-- Table structure for table `role`
--

CREATE TABLE `role` (
  `role_id` int(11) NOT NULL,
  `role_name` varchar(50) DEFAULT NULL,
  `role_folder` varchar(50) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `role`
--

INSERT INTO `role` (`role_id`, `role_name`, `role_folder`) VALUES
(1, 'admin', 'admin'),
(2, 'Viewer', 'viewer');

-- --------------------------------------------------------

--
-- Table structure for table `role_user`
--

CREATE TABLE `role_user` (
  `id_role` int(11) NOT NULL,
  `id_user` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `role_user`
--

INSERT INTO `role_user` (`id_role`, `id_user`) VALUES
(1, 1);

-- --------------------------------------------------------

--
-- Table structure for table `user`
--

CREATE TABLE `user` (
  `id` int(11) NOT NULL,
  `full_name` varchar(50) DEFAULT NULL,
  `username` varchar(100) DEFAULT NULL,
  `password` varchar(32) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `user`
--

INSERT INTO `user` (`id`, `full_name`, `username`, `password`) VALUES
(1, 'Administrator', 'admin', '21232f297a57a5a743894a0e4a801fc3');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `class`
--
ALTER TABLE `class`
  ADD PRIMARY KEY (`class_id`);

--
-- Indexes for table `data_group_info`
--
ALTER TABLE `data_group_info`
  ADD PRIMARY KEY (`id_dgroup`);

--
-- Indexes for table `data_group_split`
--
ALTER TABLE `data_group_split`
  ADD PRIMARY KEY (`id_dtrain`,`id_dgroup`);

--
-- Indexes for table `data_input`
--
ALTER TABLE `data_input`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `data_train`
--
ALTER TABLE `data_train`
  ADD PRIMARY KEY (`id_dtrain`);

--
-- Indexes for table `model_info`
--
ALTER TABLE `model_info`
  ADD PRIMARY KEY (`id_model`);

--
-- Indexes for table `model_train`
--
ALTER TABLE `model_train`
  ADD PRIMARY KEY (`id_train`,`id_model`);

--
-- Indexes for table `model_train_state`
--
ALTER TABLE `model_train_state`
  ADD PRIMARY KEY (`id_train`);

--
-- Indexes for table `role`
--
ALTER TABLE `role`
  ADD PRIMARY KEY (`role_id`);

--
-- Indexes for table `role_user`
--
ALTER TABLE `role_user`
  ADD PRIMARY KEY (`id_role`,`id_user`);

--
-- Indexes for table `user`
--
ALTER TABLE `user`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `data_group_info`
--
ALTER TABLE `data_group_info`
  MODIFY `id_dgroup` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=2;

--
-- AUTO_INCREMENT for table `model_info`
--
ALTER TABLE `model_info`
  MODIFY `id_model` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=2;

--
-- AUTO_INCREMENT for table `role`
--
ALTER TABLE `role`
  MODIFY `role_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=3;

--
-- AUTO_INCREMENT for table `user`
--
ALTER TABLE `user`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=2;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;

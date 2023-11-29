/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include "gtest/gtest.h"
#include "tenann/util/object_pool.h"
#include "tenann/util/runtime_profile.h"
#include "tenann/util/runtime_profile_macros.h"

using namespace tenann;

TEST(RuntimeProfileTest, test_time_profile) {
  auto root = std::make_unique<RuntimeProfile>("root");
  auto child1 = std::make_unique<RuntimeProfile>("child1");
  root->add_child(child1.get(), true, nullptr);
  root->add_info_string("test", "myinfo");

  auto total = T_ADD_TIMER(root, "TotalTime");
  auto timer1 = T_ADD_TIMER(root, "Test1");
  auto timer2 = T_ADD_TIMER(root, "Test2");
  auto child_timer1 = T_ADD_TIMER(child1, "Child1::Test1");
  auto child_timer2 = T_ADD_TIMER(child1, "Child1::Test2");

  std::vector<int> values;
  {
    T_SCOPED_TIMER(total);
    {
      T_SCOPED_TIMER(timer1);
      T_SCOPED_TIMER(child_timer1);

      for (int i = 0; i < 100000; i++) {
        values.push_back(i);
      }
    }

    {
      T_SCOPED_TIMER(timer2);
      T_SCOPED_TIMER(child_timer2);

      for (int i = 0; i < 100000; i++) {
        values.push_back(i);
      }
    }
  }

  root->compute_time_in_profile();
  root->pretty_print(&std::cerr);
}